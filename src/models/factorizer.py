import os
import torch
import torch.nn.functional as F

from hydra.utils import instantiate
from omegaconf import DictConfig

from .base_model import Model
from .classifier import Classifier
from src.utils.datasets import HomerData
from src.utils.config import get_prev_config


class Factorizer(Model):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)
        self.net1 = self.net
        self.net2 = instantiate(cfg.net2)
        self.log_acc_sim = cfg.dataset.log_acc_sim
        self.learn_acc = cfg.learn_acc
        self.smear = cfg.smear
        self.iterate_target = cfg.iterate_target

        if self.smear:
            self.smearing_kernel = torch.distributions.Normal(
                loc=0.0, scale=cfg.smear_kernel_scale
            )

    def batch_loss(self, batch: HomerData) -> torch.Tensor:

        if self.iterate_target and batch.w_ref_event is not None:
            batch.w_class *= batch.w_ref_event

        # forward pass
        g1, g2 = self.forward(batch.breaks)
        log_w = g1 - g2
        log_W = (log_w * batch.accepted).sum(1)

        if self.learn_acc:
            # calculate acceptance efficiencies (w/ correction due to iterations)
            log_acc_sim = self.log_acc_sim
            if not self.iterate_target and batch.w_ref_break is not None:
                log_w_sample = batch.w_ref_break.log()
                # correct a_inf
                log_w = log_w + log_w_sample
                # correct a_sim
                acc_sim = self.inferred_acceptance(
                    log_w_sample, batch.num_rej, batch.in_chain_n
                )
                log_acc_sim = acc_sim.log()
            acc_inf = self.inferred_acceptance(log_w, batch.num_rej, batch.in_chain_n)
            log_W = log_W - acc_inf.log() + log_acc_sim

        if self.smear:
            # smear weights with feature-space distances
            x = batch.observables  # (batch, features)
            chain_weights = self.smearing_kernel.log_prob(torch.cdist(x, x)).exp()
            log_W = (chain_weights @ log_W.exp()).log() - chain_weights.sum(1).log()

        # L_C (BCE)
        loss_c = (batch.w_class + 1) * F.softplus(-log_W) + log_W

        # L_12 (BCE)
        loss_12 = (g1.exp() + 1) * F.softplus(-g2) + g2

        # include sample weights
        if not self.iterate_target and batch.w_ref_event is not None:
            loss_c = loss_c * batch.w_ref_event
            loss_12 = loss_12 * batch.w_ref_break

        loss_c = loss_c.mean(0)
        loss_12 = ((loss_12 * batch.is_break).sum(1) / batch.is_break.sum(1)).mean(0)
        loss = loss_c + loss_12

        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        # log quantities to tensorboard
        with torch.no_grad():
            self.log_scalar(loss_c, "loss_c")
            self.log_scalar(loss_12, "loss_12")
            self.log_scalar((g1 * batch.is_break).sum(1).mean(0), "g1")
            self.log_scalar((g2 * batch.is_break).sum(1).mean(0), "g2")

            if self.learn_acc:
                self.log_scalar(acc_inf, "acc_inf")

        return loss

    @staticmethod
    def inferred_acceptance(
        log_break_weights: torch.Tensor,
        num_rej: torch.Tensor,
        in_chain_n: torch.Tensor,
    ) -> torch.Tensor:

        device = log_break_weights.device

        # calculate weight for each chain, indexed by n
        in_chain_n = in_chain_n.permute(2, 0, 1)
        weight_chain_n = (log_break_weights * in_chain_n).sum(2).exp()
        weight_chain_n = weight_chain_n * in_chain_n.any(2)  # remove empty chains

        # calculate cumulative weights
        batch_idx = torch.arange(len(num_rej), device=device)
        weight_acc = weight_chain_n[num_rej.int(), batch_idx]
        weight_tot = weight_chain_n.sum(0)

        return weight_acc.sum(0) / weight_tot.sum(0)

    def reseed(self):
        if self.bayesian:
            self.net.reseed()
            self.net2.reseed()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate the networks"""
        g1 = self.net(s[..., :7]).squeeze(-1)
        g2 = self.net2(s[..., 5:7]).squeeze(-1)
        return g1, g2

    def log_break_weight(self, s: torch.Tensor) -> torch.Tensor:
        g1, g2 = self.forward(s)
        return g1 - g2

    def load_nets(self, model_dir: str, device: torch.device):
        model_path = os.path.join(model_dir, "model.pt")
        state_dicts = torch.load(model_path, map_location=device, weights_only=False)
        model_dict = state_dicts["model"]
        net_dict = {k[4:]: v for k, v in model_dict.items() if k.startswith("net.")}
        net2_dict = {k[5:]: v for k, v in model_dict.items() if k.startswith("net2.")}
        net3_dict = {k[5:]: v for k, v in model_dict.items() if k.startswith("net3.")}
        self.net.load_state_dict(net_dict)
        if net2_dict:
            self.net2.load_state_dict(net2_dict)
        if net3_dict:
            self.net3.load_state_dict(net3_dict)


class UncertaintiesFactorizer(Factorizer):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        self.net1 = self.net
        self.net2 = instantiate(cfg.net2)

        # load step_one classifier
        classifier_cfg = get_prev_config(cfg.w_class_path)
        self.classifier = Classifier(classifier_cfg)
        self.classifier.load(cfg.w_class_path, device=torch.device("cpu"))

    def batch_loss(self, batch: HomerData) -> torch.Tensor:

        # construct target weight
        with torch.no_grad():

            # disable dropout, batchnorm etc.
            self.classifier.eval()

            # resample classifier weights if BNN
            if self.classifier.bayesian:
                self.classifier.reseed()

            # call classifier
            # TODO: problematic for pointcloud classifier with different preprocessing
            target = self.classifier(batch)

            # get total ref to data weight
            logw_total = target.clone()
            if batch.w_ref_event is not None:
                # add reference weight to target
                logw_total += batch.w_ref_event.log()
                if self.iterate_target:
                    target = logw_total

            # correct target normalization
            if self.cfg.norm_target:
                target -= logw_total.exp().mean().log()

        # rename variables
        s = batch.breaks
        accepted = batch.accepted

        # forward pass: mean and logvar of break-level log weight
        logw_s, logvar_logw_s = self.forward(s).unbind(-1)
        logvar_logw_s = logvar_logw_s.clamp(-20, 10)

        # combine into mu, var at chain level
        logw_S = (logw_s * accepted).sum(1)  # sum of means
        var_logw_S = (logvar_logw_s.exp() * accepted).sum(1)  # sum of variances

        if self.learn_acc:
            # calculate acceptance efficiencies (w/ correction due to iterations)
            log_acc_sim = self.log_acc_sim
            if not self.iterate_target and batch.w_ref_break is not None:
                log_w_ref = batch.w_ref_break.log()
                # correct a_inf
                logw_s = logw_s + log_w_ref
                # correct a_sim
                acc_sim = self.inferred_acceptance(
                    log_w_ref, batch.num_rej, batch.in_chain_n
                )
                log_acc_sim = acc_sim.log()
            acc_inf = self.inferred_acceptance(logw_s, batch.num_rej, batch.in_chain_n)
            # correct event weight
            logw_S = logw_S + log_acc_sim - acc_inf.log()

        # Gaussian likelihood loss
        loss = F.gaussian_nll_loss(target, logw_S, var_logw_S, reduction="none")

        if not self.iterate_target and batch.w_ref_break is not None:
            loss = loss * batch.w_ref_event

        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        # tensorboard logging
        with torch.no_grad():
            self.log_scalar(loss, "nll")
            self.log_scalar(logw_S.mean(0), "logw_S")
            self.log_scalar(var_logw_S.mean(0), "var_logw_S")
            self.log_scalar((logw_s * accepted).sum() / accepted.int().sum(), "logw_s")
            self.log_scalar(
                (logvar_logw_s.exp() * accepted).sum() / accepted.int().sum(),
                "var_ln_w",
            )
            if self.learn_acc:
                self.log_scalar(acc_inf, "acc_inf")

        return loss.mean(0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Evaluate the networks"""
        # add pT_string to delta_pt
        s[..., 2:4] += s[..., 5:7]
        mu = self.net(s[..., :5])
        logvar = self.net2(s[..., :5])

        return torch.cat([mu, logvar], -1)

    def log_break_weight(self, s: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(s).unbind(-1)
        return mu, logvar.exp()
