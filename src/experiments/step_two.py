import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from collections import defaultdict
from hydra.utils import instantiate
from itertools import pairwise
from matplotlib.backends.backend_pdf import PdfPages

from src.experiments.training import TrainingExperiment
from src.utils import plotting
from src.utils.datasets import HomerData
from src.utils.utils import get_break_masks, get_last_break_mask


class StepTwoExperiment(TrainingExperiment):

    @property
    def iterating(self):
        return self.cfg.w_prev_path is not None
    
    @property
    def with_var(self):
        return "UncertaintiesFactorizer" in self.model.__class__.__name__

    def init_preprocessing(self):
        self.pcfg = self.cfg.dataset.preprocessing
        self.train_keys = [
            "hadrons_obs_only",
            "splits_for_full_hadron_info",
            "history_indexes",
        ]
        preprocessing = [
            instantiate(self.pcfg.hadrons),
            instantiate(self.pcfg.history, keep_stringends=False),
        ]
        return preprocessing

    def init_dataset(self, path_exp, path_sim):

        # read data (sim only)
        # data_device = self.device if self.cfg.data.on_gpu else torch.device("cpu")
        dset = HomerData.from_dir(
            path=path_sim,
            num=self.cfg.data.num,
            w_class_path=self.cfg.w_class_path,
            w_prev_path=self.cfg.w_prev_path,
            keys=self.train_keys,
            # device=data_device,
        ).to(self.dtype)

        return dset

    def collate_fn(self, batch: HomerData):
        """
        Calculates a mask indicating if each break is in the nth chain.
        """
        batch.num_rej = batch.history_indexes.max(1).values
        chain_index = torch.arange(batch.num_rej.max() + 1, device=batch.device)
        batch.in_chain_n = batch.history_indexes.unsqueeze(-1) == chain_index
        return batch

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the model on the given dataloader.
        Each variant of the weights are saved
        """

        # disable batchnorm updates, dropout etc.
        self.model.eval()

        # get predictions across the test set
        prediction_stack = defaultdict(list)
        n_evals = self.cfg.num_bnn_samples if self.model.bayesian else 1
        for _ in range(n_evals):

            if self.model.bayesian:  # sample new bnn weights
                self.model.reseed()

            # predict in batches
            predictions = defaultdict(list)
            for batch in dataloader:

                batch = batch.to(self.device, non_blocking=True)
                lw = self.model.log_break_weight(batch.splits_for_full_hadron_info)

                if self.with_var:
                    # unpack into mean and variance if heteroscedastic model
                    lw, var_lw = lw

                # include sample weight
                if self.iterating and not self.model.iterate_target:
                    # merge with sample weights
                    lw += batch.w_ref_break.log()

                # set weights for padded entries to NaN
                break_weight = torch.where(batch.is_break, lw.exp(), 1.0)

                if self.with_var:
                    # compute variance on the chain and history log weights
                    var_lW = (var_lw * batch.accepted).sum(1)
                    var_lH = (var_lw * batch.is_break).sum(1)

                # include correction for acceptance efficiencies
                acc_inf = self.model.inferred_acceptance(
                    lw, batch.num_rej, batch.in_chain_n
                )

                # calculate homer weight and chain weight
                homer_weight = (lw * batch.is_break).sum(1).exp()
                chain_weight = (lw * batch.accepted).sum(1).exp()

                acc_sim = np.exp(self.cfg.dataset.log_acc_sim)
                if self.cfg.learn_acc:
                    event_weight = chain_weight * acc_sim / acc_inf
                else:
                    event_weight = chain_weight.clone()
                    chain_weight *= acc_inf / acc_sim
                    homer_weight *= acc_inf / acc_sim

                if self.cfg.smear:
                    self.log.info("Computing smeared homer weight")
                    smear_weight = homer_weight.clone()
                    for i, weight in enumerate(smear_weight):
                        kernel_weight = (
                            self.model.smearing_kernel.log_prob(
                                torch.cdist(
                                    batch.hadrons_obs_only[[i]],
                                    batch.hadrons_obs_only,
                                )
                            )
                            .exp()
                            .squeeze(0)
                        )
                        smear_weight[i] = (kernel_weight * weight).sum(
                            0
                        ) / kernel_weight.sum(0)

                # append results
                predictions["break_weights"].append(break_weight.cpu().numpy())
                predictions["chain_weights"].append(chain_weight.cpu().numpy())
                predictions["homer_weights"].append(homer_weight.cpu().numpy())
                predictions["event_weights"].append(event_weight.cpu().numpy())

                if self.with_var:
                    predictions["vars_log_break_weight"].append(var_lw.cpu().numpy())
                    predictions["vars_log_chain_weight"].append(var_lW.cpu().numpy())
                    predictions["vars_log_homer_weight"].append(var_lH.cpu().numpy())

                if self.cfg.smear:
                    predictions["smear_weights"].append(smear_weight.cpu().numpy())

            # stack batches
            for k in predictions:
                prediction_stack[k].append(np.concatenate(predictions[k]))

        # stack evaluations
        for k in prediction_stack:
            prediction_stack[k] = np.stack(prediction_stack[k])

        tag = "" if tag is None else f"_{tag}"
        savepath = os.path.join(self.exp_dir, f"predictions{tag}.npz")
        self.log.info(f"Saving labels and weights to {savepath}")
        np.savez(savepath, **prediction_stack)

    def plot(self):

        pcfg = self.cfg.plotting
        pw = pcfg.pagewidth

        savedir = os.path.join(self.exp_dir, "plots")
        os.makedirs(savedir, exist_ok=True)

        # read predicted weights from disk
        self.log.info("Reading weights from disk")
        record = np.load(os.path.join(self.exp_dir, "predictions.npz"))
        break_weights = record["break_weights"]
        chain_weights = record["chain_weights"]
        homer_weights = record["homer_weights"]
        event_weights = (
            record["smear_weights"] if self.cfg.smear else record["event_weights"]
        )  # TODO: Check if this is valid
        break_vars = record["vars_log_break_weight"] if self.with_var else None
        chain_vars = record["vars_log_chain_weight"] if self.with_var else None
        homer_vars = record["vars_log_homer_weight"] if self.with_var else None
        try:
            event_vars = record["vars_log_event_weight"]
        except KeyError:
            event_vars = chain_vars

        # load test data for histograms
        self.log.info("Loading test data")
        plot_keys = [
            "hadrons_obs_only",
            "history_indexes",
            "splits_for_full_hadron_info",
            "analytical_per_split_log_weight",
        ]
        exp = HomerData.from_dir(
            path=self.cfg.data.path_exp_test,
            w_class_path=self.cfg.w_class_path,
            w_prev_path=self.cfg.w_prev_path,
            keys=plot_keys,
            num=self.cfg.data.num,
        )
        sim = HomerData.from_dir(
            path=self.cfg.data.path_sim_test,
            w_class_path=self.cfg.w_class_path,
            w_prev_path=self.cfg.w_prev_path,
            keys=plot_keys,
            num=self.cfg.data.num,
        )

        # extract sample weights
        exp_weights = exp.sample_weights.numpy()

        # get masks
        is_break_exp, accepted_exp, in_chain_n_exp = get_break_masks(exp)
        is_break_sim, accepted_sim, in_chain_n_sim = get_break_masks(sim)
        is_break_exp = is_break_exp.numpy()
        is_break_sim = is_break_sim.numpy()
        accepted_exp = accepted_exp.numpy()
        accepted_sim = accepted_sim.numpy()
        in_chain_n_exp = in_chain_n_exp.numpy()
        in_chain_n_sim = in_chain_n_sim.numpy()

        # compute exact weights
        exact_break_weights_exp = exp.analytical_per_split_log_weight.exp().numpy()
        exact_break_weights_sim = sim.analytical_per_split_log_weight.exp().numpy()
        exact_chain_weights_exp = torch.exp(
            (exp.analytical_per_split_log_weight * accepted_exp).sum(1)
        ).numpy()
        exact_chain_weights_sim = torch.exp(
            (sim.analytical_per_split_log_weight * accepted_sim).sum(1)
        ).numpy()
        exact_history_weights_exp = torch.exp(
            (exp.analytical_per_split_log_weight * is_break_exp).sum(1)
        ).numpy()
        exact_history_weights_sim = torch.exp(
            (sim.analytical_per_split_log_weight * is_break_sim).sum(1)
        ).numpy()
        exact_event_weights_exp = exact_chain_weights_exp * np.exp(
            self.cfg.dataset.log_acc_sim - self.cfg.dataset.log_acc_exp
        )
        exact_event_weights_sim = exact_chain_weights_sim * np.exp(
            self.cfg.dataset.log_acc_sim - self.cfg.dataset.log_acc_exp
        )

        # load classifier weights from step one
        w_class_exp = exp.w_class.numpy()
        w_class_sim = sim.w_class.numpy()

        if self.iterating:
            # define classifier weight as sample weight and ref -> data correction
            w_class_sim = w_class_sim * sim.w_ref_event[None, :].numpy()

        # plot observables
        self.log.info("Plotting event observables")
        obs_exp = exp.hadrons_obs_only.nan_to_num().numpy()
        obs_sim = sim.hadrons_obs_only.nan_to_num().numpy()
        with PdfPages(os.path.join(savedir, f"observables.pdf")) as pdf:
            for i in range(13):
                fig, ax = plotting.plot_reweighting(
                    exp=obs_exp[:, i],
                    sim=obs_sim[:, i],
                    weights_list=[
                        exact_event_weights_sim,
                        event_weights,
                        # w_class_sim,
                    ],
                    variance_list=[None, event_vars, None],
                    names_list=[
                        "(Exact Chain)",
                        "(Infer Chain)",
                        "Classifier",
                    ],
                    xlabel=pcfg.hadron_obs_labels[i],
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=40,
                    discrete=1 if i == 5 else 2 if i == 6 else False,
                    logy=True,
                    denom_idx=2,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

            # TODO: Need w_class from iteration 1 to run this
            # # classifier optimal observable
            # fig, ax = plotting.plot_reweighting(
            #     exp=-2 * np.log(w_class_exp),
            #     sim=-2 * np.log(w_class_sim),
            #     weights_list=[
            #         w_class_sim,
            #         event_weights,
            #         # homer_weights,
            #         exact_event_weights_sim,
            #         # exact_history_weights_sim,
            #     ],
            #     variance_list=[
            #         None,
            #         event_vars,
            #         # homer_vars,
            #         None,
            #         # None
            #     ],
            #     names_list=[
            #         "Classifier",
            #         "(Infer Chain)",
            #         # "(Infer History)",
            #         "(Exact Chain)",
            #         # "(Exact History)",
            #     ],
            #     xlabel=r"$-2\log w_\mathrm{class}(x)$",
            #     figsize=np.array([1, 5 / 6]) * pw / 2,
            #     num_bins=40,
            #     discrete=False,
            #     logy=False,
            #     # ratio_lims=(0.5, 1.5),
            # )
            # pdf.savefig(fig)
            # plt.close(fig)

            self.log.info("Plotting optimal observables")
            # exact optimal chain observable
            fig, ax = plotting.plot_reweighting(
                exp=-2 * np.log(exact_chain_weights_exp),
                sim=-2 * np.log(exact_chain_weights_sim),
                weights_list=[
                    exact_event_weights_sim,
                    event_weights,
                    # w_class_sim,
                ],
                variance_list=[
                    None,
                    chain_vars,
                    None,
                ],
                names_list=[
                    "(Exact Chain)",
                    "(Infer Chain)",
                    "Classifier",
                ],
                xlabel=r"$-2\log w_\mathrm{exact}(S)$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                discrete=False,
                logy=False,
                denom_idx=2,
                # ratio_lims=(0.5, 1.5),
                exp_weights=exp_weights,
            )
            pdf.savefig(fig)
            plt.close(fig)

            # exact optimal history observable
            fig, ax = plotting.plot_reweighting(
                exp=-2 * np.log(exact_history_weights_exp),
                sim=-2 * np.log(exact_history_weights_sim),
                weights_list=[
                    exact_history_weights_sim,
                    homer_weights,
                    # w_class_sim,
                ],
                variance_list=[
                    None,
                    homer_vars,
                    None,
                ],
                names_list=[
                    "(Exact History)",
                    "(Infer History)",
                    "Classifier",
                ],
                xlabel=r"$-2\log w_\mathrm{exact}(H)$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                discrete=False,
                logy=False,
                ratio_lims=(0.5, 1.5),
                denom_idx=2,
                exp_weights=exp_weights,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # fragmentation
        with PdfPages(os.path.join(savedir, f"fragmentation.pdf")) as pdf:

            frg_exp = exp.splits_for_full_hadron_info.numpy()
            frg_sim = sim.splits_for_full_hadron_info.numpy()

            # average
            self.log.info("Plotting marginal fragmentation function")
            self.log.info(
                f"Mean weight of all emission: "
                f"{break_weights[..., is_break_sim].mean()}"
            )
            fig, ax = plotting.plot_reweighting(
                exp=frg_exp[is_break_exp, 0],
                sim=frg_sim[is_break_sim, 0],
                weights_list=[
                    exact_break_weights_sim[is_break_sim],
                    break_weights[..., is_break_sim],
                ],
                variance_list=[
                    None,
                    None if break_vars is None else break_vars[..., is_break_sim],
                ],
                names_list=["Exact", "Inferred"],
                xlabel=r"$z$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                density=True,
                qlims=(0.0, 0.995),
                title="Inclusive",
                denom_idx=2,
                exp_weights=exp_weights.repeat(100).reshape(-1, 100)[is_break_exp].T,
            )
            pdf.savefig(fig)
            plt.close(fig)

            # fixed mThad2
            mThad2_exp = frg_exp[..., 3] ** 2 + (
                (frg_exp[..., 1:3] + frg_exp[..., 5:7]) ** 2
            ).sum(-1)
            mThad2_sim = frg_sim[..., 3] ** 2 + (
                (frg_sim[..., 1:3] + frg_sim[..., 5:7]) ** 2
            ).sum(-1)

            mThad2_quantiles = np.array(
                [np.quantile(mThad2_sim[is_break_sim], i * 10 / 100) for i in range(11)]
            )
            self.log.info("Plotting conditional fragmentation function")
            for i, (lo, hi) in enumerate(pairwise(mThad2_quantiles)):

                mask_exp = is_break_exp & (mThad2_exp > lo) & (mThad2_exp < hi)
                mask_sim = is_break_sim & (mThad2_sim > lo) & (mThad2_sim < hi)
                self.log.info(
                    f"Mean weight of {i}. bin: "
                    f"{np.nanmean([break_weights[..., mask_sim]])}",
                )
                fig, ax = plotting.plot_reweighting(
                    exp=frg_exp[mask_exp, 0],
                    sim=frg_sim[mask_sim, 0],
                    weights_list=[
                        exact_break_weights_sim[mask_sim],
                        break_weights[..., mask_sim],
                    ],
                    variance_list=[
                        None,
                        None if break_vars is None else break_vars[..., mask_sim],
                    ],
                    names_list=["Exact", "Inferred"],
                    xlabel=r"$z$",
                    title=rf"{lo:.3f} < $m_T^2$ < {hi:.3f}",
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=40,
                    density=True,
                    qlims=(0.0, 0.995),
                    denom_idx=2,
                    exp_weights=exp_weights.repeat(100).reshape(-1, 100)[mask_exp].T,
                )
                pdf.savefig(fig)
                plt.close(fig)

            self.log.info("Plotting marginal transverse mass")
            fig, ax = plotting.plot_reweighting(
                exp=np.sqrt(mThad2_exp)[is_break_exp],
                sim=np.sqrt(mThad2_sim)[is_break_sim],
                weights_list=[
                    exact_break_weights_sim[is_break_sim],
                    break_weights[..., is_break_sim],
                ],
                variance_list=[
                    None,
                    None if break_vars is None else break_vars[..., is_break_sim],
                ],
                names_list=["Exact", "Inferred"],
                xlabel=r"$m_T$",
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=40,
                density=True,
                qlims=(0.0, 0.995),
                title="Inclusive",
                denom_idx=2,
                exp_weights=exp_weights.repeat(100).reshape(-1, 100)[is_break_exp].T,
            )
            pdf.savefig(fig)
            plt.close(fig)

            z_quantiles = np.array(
                [
                    np.quantile(frg_sim[..., 0][is_break_sim], i * 10 / 100)
                    for i in range(11)
                ]
            )
            self.log.info("Plotting conditional transverse mass")
            for i, (lo, hi) in enumerate(pairwise(z_quantiles)):
                mask_exp = (
                    is_break_exp & (frg_exp[..., 0] > lo) & (frg_exp[..., 0] < hi)
                )
                mask_sim = (
                    is_break_sim & (frg_sim[..., 0] > lo) & (frg_sim[..., 0] < hi)
                )
                self.log.info(
                    f"Mean weight of {i}. bin: "
                    f"{np.nanmean([break_weights[..., mask_sim]])}"
                )
                fig, ax = plotting.plot_reweighting(
                    exp=np.sqrt(mThad2_exp)[mask_exp],
                    sim=np.sqrt(mThad2_sim)[mask_sim],
                    weights_list=[
                        exact_break_weights_sim[mask_sim],
                        break_weights[..., mask_sim],
                    ],
                    variance_list=[
                        None,
                        None if break_vars is None else break_vars[..., mask_sim],
                    ],
                    names_list=["Exact", "Inferred"],
                    xlabel=r"$m_T$",
                    title=rf"{lo:.3f} < $z$ < {hi:.3f}",
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=40,
                    density=True,
                    qlims=(0.0, 0.995),
                    denom_idx=2,
                    exp_weights=exp_weights.repeat(100).reshape(-1, 100)[mask_exp].T,
                )
                pdf.savefig(fig)
                plt.close(fig)

            # # per-emission
            # self.log.info("Plotting per-emission fragmentation function")
            # for i in range(5):
            #     self.log.info(
            #         f"Mean weight in {i}. emission: "
            #         f"{break_weights[..., in_chain_n_sim[i]].mean()}"
            #     )
            #     fig, ax = plotting.plot_reweighting(
            #         exp=frg_exp[in_chain_n_exp[i], 0],
            #         sim=frg_sim[in_chain_n_sim[i], 0],
            #         weights_list=[
            #             exact_break_weights_sim[in_chain_n_sim[i]],
            #             break_weights[..., in_chain_n_sim[i]],
            #         ],
            #         variance_list=[
            #             None,
            #             (
            #                 None
            #                 if break_vars is None
            #                 else break_vars[..., in_chain_n_sim[i]]
            #             ),
            #         ],
            #         names_list=["Exact", "Inferred"],
            #         xlabel="$z$",
            #         title=f"Emission {i+1}",
            #         figsize=np.array([1, 5 / 6]) * pw / 2,
            #         num_bins=40,
            #         density=True,
            #         denom_idx=2,
            #         exp_weights=exp_weights.repeat(100)
            #         .reshape(-1, 100)[in_chain_n_exp[i]]
            #         .T,
            #     )
            #     pdf.savefig(fig)
            #     plt.close(fig)

            # I_exp = exp.history_indexes
            # I_sim = sim.history_indexes
            # # TODO: potential problem with accepted n=1 due to updating history_indexes in get_break mask?
            # last_break_acc_exp = accepted_exp & get_last_break_mask(I_exp, n=1).numpy()
            # last_break_acc_sim = accepted_sim & get_last_break_mask(I_sim, n=1).numpy()
            # last_break_rej_exp = ~accepted_exp & get_last_break_mask(I_exp, n=1).numpy()
            # last_break_rej_sim = ~accepted_sim & get_last_break_mask(I_sim, n=1).numpy()
            # self.log.info(
            #     f"Mean weight of last accepted emissions: "
            #     f"{np.nanmean(break_weights[..., last_break_acc_sim])}"
            # )
            # fig, ax = plotting.plot_reweighting(
            #     exp=frg_exp[last_break_acc_exp, 0],
            #     sim=frg_sim[last_break_acc_sim, 0],
            #     weights_list=[
            #         exact_break_weights_sim[last_break_acc_sim],
            #         break_weights[..., last_break_acc_sim],
            #     ],
            #     variance_list=[
            #         None,
            #         (
            #             None
            #             if break_vars is None
            #             else break_vars[..., last_break_acc_sim]
            #         ),
            #     ],
            #     names_list=["Exact", "Inferred"],
            #     xlabel="$z$",
            #     title=f"Last accepted emission",
            #     figsize=np.array([1, 5 / 6]) * pw / 2,
            #     num_bins=40,
            #     density=True,
            #     denom_idx=2,
            #     exp_weights=exp_weights.repeat(100)
            #     .reshape(-1, 100)[last_break_acc_exp]
            #     .T,
            # )
            # pdf.savefig(fig)
            # plt.close(fig)

            # self.log.info(
            #     f"Mean weight of last rejected emissions: "
            #     f"{np.nanmean(break_weights[..., last_break_rej_sim])}"
            # )
            # fig, ax = plotting.plot_reweighting(
            #     exp=frg_exp[last_break_rej_exp, 0],
            #     sim=frg_sim[last_break_rej_sim, 0],
            #     weights_list=[
            #         exact_break_weights_sim[last_break_rej_sim],
            #         break_weights[..., last_break_rej_sim],
            #     ],
            #     variance_list=[
            #         None,
            #         (
            #             None
            #             if break_vars is None
            #             else break_vars[..., last_break_rej_sim]
            #         ),
            #     ],
            #     names_list=["Exact", "Inferred"],
            #     xlabel="$z$",
            #     title=f"Last rejected emission",
            #     figsize=np.array([1, 5 / 6]) * pw / 2,
            #     num_bins=40,
            #     density=True,
            #     denom_idx=2,
            #     exp_weights=exp_weights.repeat(100)
            #     .reshape(-1, 100)[last_break_rej_exp]
            #     .T,
            # )
            # pdf.savefig(fig)
            # plt.close(fig)

        # learned sigma calibration
        if self.with_var:
            with PdfPages(os.path.join(savedir, f"sigma_calibration.pdf")) as pdf:

                self.log.info("Plotting break-level sigma calibration")
                mask = is_break_sim
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(break_weights[0][mask]),
                    pred_logweight_vars=break_vars[0][mask],
                    ref_logweights=np.log(exact_break_weights_sim[mask]),
                    pred_weight_label=r"w_\phi(s)",
                    ref_weight_label=r"w_\text{exact}(s)",
                    pull_label=r"t_\text{syst}(s)",
                    ref_lims=(-0.4, 1.0),
                    diff_lims=(-0.1, 0.1),
                )
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(break_weights[0][mask]),
                    pred_logweight_vars=break_vars[0][mask],
                    ref_logweights=np.log(exact_break_weights_sim[mask]),
                    pred_weight_label=r"w_\phi(s)",
                    ref_weight_label=r"w_\text{exact}(s)",
                    pull_label=r"t_\text{syst}(s)",
                    quantile_obs=np.log(exact_break_weights_sim[mask]),
                    num_quantiles=4,
                    ref_lims=(-0.4, 1.0),
                    diff_lims=(-0.1, 0.1),
                )
                pdf.savefig(fig)
                plt.close(fig)

                self.log.info("Plotting chain-level sigma calibration")
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(chain_weights[0]),
                    pred_logweight_vars=chain_vars[0],
                    ref_logweights=np.log(exact_chain_weights_sim),
                    pred_weight_label=r"w_\phi(S)",
                    ref_weight_label=r"w_\text{exact}(S)",
                    pull_label=r"t_\text{syst}(S)",
                )
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(chain_weights[0]),
                    pred_logweight_vars=chain_vars[0],
                    ref_logweights=np.log(exact_chain_weights_sim),
                    pred_weight_label=r"w_\phi(S)",
                    ref_weight_label=r"w_\text{exact}(S)",
                    pull_label=r"t_\text{syst}(S)",
                    num_quantiles=4,
                )
                pdf.savefig(fig)
                plt.close(fig)

                self.log.info("Plotting accepted chain-level sigma calibration")
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(event_weights[0]),
                    pred_logweight_vars=event_vars[0],
                    ref_logweights=np.log(exact_event_weights_sim),
                    pred_weight_label=r"w_\phi(S_\mathrm{acc})",
                    ref_weight_label=r"w_\text{exact}(S_\mathrm{acc})",
                    pull_label=r"t_\text{syst}(S_\mathrm{acc})",
                )
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(event_weights[0]),
                    pred_logweight_vars=event_vars[0],
                    ref_logweights=np.log(exact_event_weights_sim),
                    pred_weight_label=r"w_\phi(S_\mathrm{acc})",
                    ref_weight_label=r"w_\text{exact}(S_\mathrm{acc})",
                    pull_label=r"t_\text{syst}(S_\mathrm{acc})",
                    num_quantiles=4,
                )
                pdf.savefig(fig)
                plt.close(fig)

                self.log.info("Plotting history-level sigma calibration")
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(homer_weights[0]),
                    pred_logweight_vars=homer_vars[0],
                    ref_logweights=np.log(exact_history_weights_sim),
                    pred_weight_label=r"w_\phi(H)",
                    ref_weight_label=r"w_\text{exact}(H)",
                    pull_label=r"t_\text{syst}(H)",
                )
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plotting.plot_syst_calibration(
                    pred_logweights=np.log(homer_weights[0]),
                    pred_logweight_vars=homer_vars[0],
                    ref_logweights=np.log(exact_history_weights_sim),
                    pred_weight_label=r"w_\phi(H)",
                    ref_weight_label=r"w_\text{exact}(H)",
                    pull_label=r"t_\text{syst}(H)",
                    num_quantiles=4,
                )
                pdf.savefig(fig)
                plt.close(fig)

        if w_class_sim.ndim > 1:
            w_class_exp = w_class_exp.mean(0)
            w_class_sim = w_class_sim.mean(0)
        if break_weights.ndim > 2:
            break_weights = break_weights.mean(0)
            event_weights = event_weights.mean(0)
            homer_weights = homer_weights.mean(0)

        # weights
        self.log.info("Plotting weight histograms")
        with PdfPages(os.path.join(savedir, f"weights.pdf")) as pdf:

            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-2, 2, 40)
            plotting.add_histogram(
                ax,
                exact_history_weights_sim,
                bins=bins,
                label="Exact (History)",
                color="C0",
            )
            plotting.add_histogram(
                ax,
                exact_event_weights_sim,
                bins=bins,
                label="Exact (Event)",
                color="C1",
            )
            plotting.add_histogram(
                ax, w_class_sim, bins=bins, label="Classifier", color="C2"
            )
            plotting.add_histogram(
                ax, event_weights, bins=bins, label="Inferred", color="C3"
            )
            plotting.add_histogram(
                ax, homer_weights, bins=bins, label="Homer", color="C4"
            )

            ax.set_xlabel(f"$w(x)$")
            ax.semilogx()
            ax.semilogy()
            ax.set_ylim(1, None)
            ax.legend(frameon=False, handlelength=1.4)
            fig.tight_layout(pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

            for i in range(3):
                fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
                plotting.add_histogram(
                    ax,
                    break_weights[..., in_chain_n_sim[i]],
                    bins=np.logspace(-1, 1, 40),
                    color="C0",
                )
                plotting.add_histogram(
                    ax,
                    exact_break_weights_sim[in_chain_n_sim[i]],
                    bins=np.logspace(-1, 1, 40),
                    color="C1",
                )
                ax.set_xlabel(f"$w(s_{i+1})$")
                ax.semilogx()
                ax.semilogy()
                ax.set_ylim(1, None)
                # ax.legend(frameon=False, handlelength=1.4)
                fig.tight_layout(pad=0.2)
                pdf.savefig(fig)
                plt.close(fig)

            # plot weight comparison
            self.log.info("Plotting weight correlation")
            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-1.2, 1.2, 128)

            ax.hist2d(
                event_weights, w_class_sim, bins=bins, cmap="Blues", rasterized=True
            )
            ax.set_ylabel("Classifier weight")
            ax.set_xlabel("Inferred weight")

            ax.semilogx()
            ax.semilogy()
            ax.set_aspect(1)
            ax.plot(
                [1e-5, 1e5], [1e-5, 1e5], color="#323232", alpha=0.3, lw=1.25, ls="--"
            )
            fig.tight_layout(pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-1.2, 1.2, 128)

            ax.hist2d(
                event_weights,
                exact_event_weights_sim,
                bins=bins,
                cmap="Blues",
                rasterized=True,
            )
            ax.set_ylabel("Exact event weight")
            ax.set_xlabel("Inferred event weight")

            ax.semilogx()
            ax.semilogy()
            ax.set_aspect(1)
            ax.plot(
                [1e-5, 1e5], [1e-5, 1e5], color="#323232", alpha=0.3, lw=1.25, ls="--"
            )
            fig.tight_layout(pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-1.2, 1.2, 128)

            ax.hist2d(
                homer_weights,
                exact_history_weights_sim,
                bins=bins,
                cmap="Blues",
                rasterized=True,
            )
            ax.set_ylabel("Exact history weight")
            ax.set_xlabel("Inferred history weight")

            ax.semilogx()
            ax.semilogy()
            ax.set_aspect(1)
            ax.plot(
                [1e-5, 1e5], [1e-5, 1e5], color="#323232", alpha=0.3, lw=1.25, ls="--"
            )
            fig.tight_layout(pad=0.2)
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=np.array([1, 5 / 6]) * pw / 3)
            bins = np.logspace(-1.2, 1.2, 128)

            ax.hist2d(
                homer_weights, w_class_sim, bins=bins, cmap="Blues", rasterized=True
            )
            ax.set_ylabel("Classifier weight")
            ax.set_xlabel("Homer weight")

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
