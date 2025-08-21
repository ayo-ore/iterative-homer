import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from collections import defaultdict
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils import plotting
from src.utils.datasets import HomerData
from src.utils.utils import get_break_masks


class StepOneExperiment(TrainingExperiment):

    @property
    def iterating(self):
        return self.cfg.w_prev_path is not None

    def init_preprocessing(self):

        pcfg = self.cfg.dataset.preprocessing

        if self.model.lowlevel:
            self.train_keys = ["point_cloud"]
            preprocessing = [instantiate(pcfg.point_cloud)]
        else:
            self.train_keys = ["observables"]
            preprocessing = [instantiate(pcfg.hadrons)]

        return preprocessing

    def init_dataset(self, path_exp, path_sim):

        # read data
        # data_device = self.device if self.cfg.data.on_gpu else torch.device("cpu")
        dset_exp, dset_sim = [
            HomerData.from_dir(
                path=p,
                num=self.cfg.data.num,
                w_prev_path=self.cfg.w_prev_path,
                keys=self.train_keys,
                # device=data_device,
            )
            for p in (path_exp, path_sim)
        ]

        if dset_sim.w_ref_event is not None:
            # normalize reference weights
            dset_sim.w_ref_event /= dset_sim.w_ref_event.mean()

        # stack inputs
        dset = torch.cat([dset_exp, dset_sim], dim=0).to(self.dtype)

        # create labels
        dset.labels = torch.zeros(len(dset), dtype=dset.dtype)  # , device=data_device)
        dset.labels[: len(dset_exp)] = 1

        return dset

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the Classifier on the test dataset.
        Predictions are saved alongside truth labels
        """

        self.model.eval()

        # get predictions across the test set
        predictions = defaultdict(list)
        n_evals = self.cfg.num_bnn_samples if self.model.bayesian else 1
        for _ in range(n_evals):

            if self.model.bayesian:  # sample new bnn weights
                self.model.reseed()

            # collect predictions
            batch_preds = [
                self.model(batch.to(self.device, non_blocking=True)).cpu()
                for batch in dataloader
            ]
            predictions["logits"].append(torch.cat(batch_preds))

        # stack
        predictions["logits"] = torch.stack(predictions["logits"])

        # convert
        predictions["weights"] = predictions["logits"].exp().numpy()
        predictions["probs"] = predictions["logits"].sigmoid().numpy()
        predictions["logits"] = predictions["logits"].numpy()

        # read labels from dataloaders
        predictions["labels"] = dataloader.dataset.labels.cpu().numpy()
        if self.iterating:
            predictions["w_ref"] = dataloader.dataset.w_ref_event.cpu().numpy()

        # save to disk
        tag = "" if tag is None else f"_{tag}"
        savepath = os.path.join(self.exp_dir, f"predictions{tag}.npz")
        self.log.info(f"Saving {tag} labels, weights and probs to {savepath}")
        np.savez(savepath, **predictions)

    def plot(self):

        pcfg = self.cfg.plotting
        pw = pcfg.pagewidth

        savedir = os.path.join(self.exp_dir, "plots")
        os.makedirs(savedir, exist_ok=True)

        # load test data for histograms
        self.log.info("Loading test data")
        plot_keys = [
            "observables",
            "history_indices",
            "breaks",
            "exact_break_logweights",
        ]
        exp = HomerData.from_dir(
            path=self.cfg.data.path_exp_test,
            w_prev_path=self.cfg.w_prev_path,
            keys=plot_keys,
            num=self.cfg.data.num,
        )
        sim = HomerData.from_dir(
            path=self.cfg.data.path_sim_test,
            w_prev_path=self.cfg.w_prev_path,
            keys=plot_keys,
            num=self.cfg.data.num,
        )

        # extract sample weights
        exp_weights = exp.data_weights.numpy()

        # read predicted weights from disk
        self.log.info("Reading predictions from disk")
        record = np.load(os.path.join(self.exp_dir, "predictions.npz"))
        labels, weights, probs = record["labels"], record["weights"], record["probs"]
        w_ref = record["w_ref"] if self.iterating else None

        # calculate AUC
        data_weight = np.ones(len(labels))
        if exp_weights is not None:
            data_weight[labels == 1] = exp_weights
        if w_ref is not None:
            data_weight *= w_ref
        aucs = np.array([roc_auc_score(labels, p, sample_weight=w_ref) for p in probs])
        auc_mu, auc_std = aucs.mean(), aucs.std()

        probs = probs.mean(0)

        # extract classifier weights
        w_class_sim = weights[..., labels == 0]
        w_class_exp = weights[..., labels == 1]
        if self.iterating:
            sim.w_ref_event /= (
                sim.w_ref_event.mean()
            )  # since we're just evaluating the classifier
            w_class_sim = w_class_sim * sim.w_ref_event[None, :].numpy()
            w_class_exp = (
                w_class_exp * exp.w_ref_event[None, :].numpy()
            )  # TODO: Is this doing anything? Perhaps need to pull it again from scratch

        # get masks
        self.log.info("Constructing masks")
        is_break_exp, accepted_exp, in_chain_n_exp = get_break_masks(exp)
        is_break_sim, accepted_sim, in_chain_n_sim = get_break_masks(sim)
        is_break_exp = is_break_exp.numpy()
        is_break_sim = is_break_sim.numpy()
        accepted_exp = accepted_exp.numpy()
        accepted_sim = accepted_sim.numpy()
        in_chain_n_exp = in_chain_n_exp.numpy()
        in_chain_n_sim = in_chain_n_sim.numpy()

        # exact weights
        self.log.info("Calculating exact weights")
        # exact_break_weights_exp = exp.exact_split_logweights.exp()
        # exact_break_weights_sim = sim.exact_split_logweights.exp()
        exact_event_weights_exp = (
            (exp.exact_split_logweights * accepted_exp).sum(1).exp()
        ) * np.exp(self.cfg.dataset.log_acc_sim - self.cfg.dataset.log_acc_exp)
        exact_event_weights_sim = (
            (sim.exact_split_logweights * accepted_sim).sum(1).exp()
        ) * np.exp(self.cfg.dataset.log_acc_sim - self.cfg.dataset.log_acc_exp)
        exact_history_weights_exp = (
            (exp.exact_split_logweights * is_break_exp).sum(1).exp()
        )
        exact_history_weights_sim = (
            (sim.exact_split_logweights * is_break_sim).sum(1).exp()
        )

        # observables
        self.log.info("Plotting event observables")
        obs_exp = exp.observables.nan_to_num().numpy()
        obs_sim = sim.observables.nan_to_num().numpy()
        with PdfPages(os.path.join(savedir, f"observables.pdf")) as pdf:
            for i in range(13):
                fig, ax = plotting.plot_reweighting(
                    exp=obs_exp[:, i],
                    sim=obs_sim[:, i],
                    weights_list=[exact_event_weights_sim, w_class_sim],
                    variance_list=[None, None],
                    names_list=["Exact", "Classifier"],
                    xlabel=pcfg.hadron_obs_labels[i],
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=1 if i == 5 else 2 if i == 6 else False,
                    logy=True,
                    denom_idx=2,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

            # # classifier optimal observable
            # fig, ax = plotting.plot_reweighting(
            #     exp=-2 * np.log(w_class_exp).mean(0),
            #     sim=-2 * np.log(w_class_sim).mean(0),
            #     weights_list=[w_class_sim, exact_history_weights_sim],
            #     variance_list=[None, None],
            #     names_list=["Classifier", "Exact"],
            #     xlabel=r"$-2\log w_\mathrm{class}$",
            #     figsize=np.array([1, 5 / 6]) * pw / 2,
            #     num_bins=40,
            #     discrete=False,
            #     logy=False,
            # )
            # pdf.savefig(fig)
            # plt.close(fig)

            self.log.info("Plotting optimal observables")
            # exact optimal observable
            fig, ax = plotting.plot_reweighting(
                exp=-2 * np.log(exact_event_weights_exp),
                sim=-2 * np.log(exact_event_weights_sim),
                weights_list=[exact_event_weights_sim, w_class_sim],
                variance_list=[None, None],
                names_list=["Exact", "Classifier"],
                xlabel=r"$-2\log w_\mathrm{exact}(S)$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                discrete=False,
                logy=False,
                denom_idx=2,
                exp_weights=exp_weights,
            )
            pdf.savefig(fig)
            plt.close(fig)

            # exact optimal observable
            fig, ax = plotting.plot_reweighting(
                exp=-2 * np.log(exact_history_weights_exp),
                sim=-2 * np.log(exact_history_weights_sim),
                weights_list=[exact_history_weights_sim, w_class_sim],
                variance_list=[None, None],
                names_list=["Exact", "Classifier"],
                xlabel=r"$-2\log w_\mathrm{exact}(H)$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                discrete=False,
                logy=False,
                denom_idx=2,
                exp_weights=exp_weights,
            )
            pdf.savefig(fig)
            plt.close(fig)

        if self.cfg.net.bayesian:
            # pulls
            self.log.info("Plotting event observable uncertainty pulls")
            with PdfPages(os.path.join(savedir, f"hist_pulls.pdf")) as pdf:
                for i in range(13):
                    fig, ax = plotting.plot_reweighting_pulls(
                        exp=obs_exp[:, i],
                        sim=obs_sim[:, i],
                        weights_list=[w_class_sim, exact_event_weights_sim],
                        names_list=["BNN", "Exact"],
                        title=pcfg.hadron_obs_labels[i],
                        figsize=np.array([1, 5 / 6]) * pw / 3,
                        exp_weights=exp_weights,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                self.log.info("Plotting optimal observable uncertainty pulls")
                fig, ax = plotting.plot_reweighting_pulls(
                    exp=-2 * np.log(exact_event_weights_exp),
                    sim=-2 * np.log(exact_event_weights_sim),
                    weights_list=[w_class_sim, exact_event_weights_sim],
                    # variance_list=[None, None],
                    names_list=["Classifier", "Exact"],
                    title=r"$-2\log w_\mathrm{exact}(S)$",
                    figsize=np.array([1, 5 / 6]) * pw / 3,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

                # exact optimal observable
                fig, ax = plotting.plot_reweighting_pulls(
                    exp=-2 * np.log(exact_history_weights_exp),
                    sim=-2 * np.log(exact_history_weights_sim),
                    weights_list=[w_class_sim, exact_history_weights_sim],
                    # variance_list=[None, None],
                    names_list=["Classifier", "Exact"],
                    title=r"$-2\log w_\mathrm{exact}(H)$",
                    figsize=np.array([1, 5 / 6]) * pw / 3,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

            # calibration
            self.log.info("Plotting event observable uncertainty calibration")
            with PdfPages(os.path.join(savedir, f"hist_calibration.pdf")) as pdf:
                for i in range(13):
                    fig, ax = plotting.plot_reweighting_calibration(
                        exp=obs_exp[:, i],
                        sim=obs_sim[:, i],
                        weights_list=[w_class_sim, exact_event_weights_sim],
                        names_list=["BNN", "Exact"],
                        title=pcfg.hadron_obs_labels[i],
                        figsize=np.array([1, 5 / 6]) * pw / 3,
                        exp_weights=exp_weights,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                self.log.info("Plotting optimal observable uncertainty calibration")
                fig, ax = plotting.plot_reweighting_calibration(
                    exp=-2 * np.log(exact_event_weights_exp),
                    sim=-2 * np.log(exact_event_weights_sim),
                    weights_list=[w_class_sim, exact_event_weights_sim],
                    # variance_list=[None, None],
                    names_list=["BNN", "Exact"],
                    title=r"$-2\log w_\mathrm{exact}(S)$",
                    figsize=np.array([1, 5 / 6]) * pw / 3,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

                # exact optimal observable
                fig, ax = plotting.plot_reweighting_calibration(
                    exp=-2 * np.log(exact_history_weights_exp),
                    sim=-2 * np.log(exact_history_weights_sim),
                    weights_list=[w_class_sim, exact_history_weights_sim],
                    # variance_list=[None, None],
                    names_list=["BNN", "Exact"],
                    title=r"$-2\log w_\mathrm{exact}(H)$",
                    figsize=np.array([1, 5 / 6]) * pw / 3,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # fragmentation
        with PdfPages(os.path.join(savedir, f"fragmentation.pdf")) as pdf:

            frg_exp = exp.breaks.numpy()
            frg_sim = sim.breaks.numpy()

            apply_mask = lambda w, m: np.expand_dims(w, 1).T.repeat(100, 1)[m].T

            # average
            self.log.info("Plotting marginal fragmentation function")
            fig, ax = plotting.plot_reweighting(
                exp=frg_exp[is_break_exp, 0],
                sim=frg_sim[is_break_sim, 0],
                weights_list=[apply_mask(w_class_sim, is_break_sim)],
                variance_list=[None],
                names_list=["Classifier"],
                xlabel="$z$",
                title="Inclusive",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                density=True,
                exp_weights=exp_weights.repeat(100).reshape(-1, 100)[is_break_exp].T,
            )
            pdf.savefig(fig)
            plt.close(fig)

            # fixed mThad2
            self.log.info("Plotting conditional fragmentation function")
            mThad2_exp = frg_exp[..., 3] ** 2 + (
                (frg_exp[..., 1:3] + frg_exp[..., 5:7]) ** 2
            ).sum(-1)
            mThad2_sim = frg_sim[..., 3] ** 2 + (
                (frg_sim[..., 1:3] + frg_sim[..., 5:7]) ** 2
            ).sum(-1)
            mask_exp = is_break_exp & (mThad2_exp > 0.063) & (mThad2_exp < 0.09)
            mask_sim = is_break_sim & (mThad2_sim > 0.063) & (mThad2_sim < 0.09)
            fig, ax = plotting.plot_reweighting(
                exp=frg_exp[mask_exp, 0],
                sim=frg_sim[mask_sim, 0],
                weights_list=[apply_mask(w_class_sim, mask_sim)],
                variance_list=[None],
                names_list=["Classifier"],
                xlabel="$z$",
                title=r"$0.063 < m_{T,\mathrm{had}}^2 < 0.09$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                density=True,
                exp_weights=exp_weights.repeat(100).reshape(-1, 100)[mask_exp].T,
            )
            pdf.savefig(fig)
            plt.close(fig)

            # per-emission
            self.log.info("Plotting per-emission fragmentation function")
            for i in range(5):
                fig, ax = plotting.plot_reweighting(
                    exp=frg_exp[in_chain_n_exp[i], 0],
                    sim=frg_sim[in_chain_n_sim[i], 0],
                    weights_list=[apply_mask(w_class_sim, in_chain_n_sim[i])],
                    variance_list=[None],
                    names_list=["Classifier"],
                    xlabel="$z$",
                    title=f"Emission {i+1}",
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=40,
                    density=True,
                    exp_weights=exp_weights.repeat(100)
                    .reshape(-1, 100)[in_chain_n_exp[i]]
                    .T,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # plot weights and calibration
        with PdfPages(os.path.join(savedir, f"classifier.pdf")) as pdf:

            # probabilities
            self.log.info("Plotting classifier probabilities")
            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.linspace(0, 1, pcfg.num_bins)
            plotting.add_histogram(
                ax,
                probs[labels == 1],
                bins=bins,
                label="Data",
                color="C0",
                weights=exp_weights,
            )
            plotting.add_histogram(
                ax, probs[labels == 0], bins=bins, label="Sim", color="C1"
            )
            ax.legend(frameon=False, handlelength=1.4)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Events")
            ax.text(
                0.03,
                0.97,
                rf"AUC = ${auc_mu:.4f}\pm {auc_std:.4f}$",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
            # ax.semilogy()
            fig.subplots_adjust(top=0.91, bottom=0.2, left=0.22, right=0.96)
            pdf.savefig(fig)
            plt.close(fig)

            # calibration
            self.log.info("Plotting classifier calibration")
            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            probs_true, probs_pred = calibration_curve(labels, probs, n_bins=50)
            ax.plot(probs_true, probs_true, "k")
            ax.plot(probs_true, probs_pred)
            ax.set_xlabel("True frequency")
            ax.set_ylabel("Predicted frequency")
            fig.subplots_adjust(top=0.91, bottom=0.2, left=0.22, right=0.96)
            pdf.savefig(fig)
            plt.close(fig)

            # weight histograms
            self.log.info("Plotting weight histograms")
            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-2, 2, pcfg.num_bins)
            plotting.add_histogram(
                ax, exact_event_weights_sim, bins=bins, label="Exact Event", color="C0"
            )
            plotting.add_histogram(
                ax,
                exact_history_weights_sim,
                bins=bins,
                label="Exact History",
                color="C1",
            )
            plotting.add_histogram(
                ax, w_class_sim.mean(0), bins=bins, label="Classifier", color="C2"
            )

            ax.set_xlabel(f"$w(x)$")
            ax.semilogx()
            ax.semilogy()
            ax.set_ylim(1, None)
            ax.legend(frameon=False, handlelength=1.4)
            fig.subplots_adjust(top=0.91, bottom=0.2, left=0.22, right=0.96)
            pdf.savefig(fig)
            plt.close(fig)

            # plot weight comparison
            self.log.info("Plotting weight correlations")
            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-1.2, 1.2, 128)

            ax.hist2d(
                w_class_sim.mean(0) if self.cfg.net.bayesian else w_class_sim.squeeze(),
                exact_event_weights_sim,
                bins=bins,
                cmap="Blues",
            )
            ax.set_xlabel("Classifier weight")
            ax.set_ylabel("Exact weight")
            ax.semilogx()
            ax.semilogy()
            ax.set_aspect(1)
            ax.plot(
                [1e-5, 1e5], [1e-5, 1e5], color="#323232", alpha=0.3, lw=1.25, ls="--"
            )
            fig.tight_layout(pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

        self.log.info(f"Saved plots to {savedir}")
