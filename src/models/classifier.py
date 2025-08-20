import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.networks import TransformerEncoder
from src.utils.datasets import HomerData


class Classifier(Model):

    @property
    def lowlevel(self):
        return isinstance(self.net, TransformerEncoder)

    def batch_loss(self, batch: HomerData) -> torch.Tensor:
        logits = self.forward(batch)
        match self.cfg.loss:
            case "bce":
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch.labels, reduction="none"
                )
            case "mlc":
                sign = 1 - 2 * batch.labels
                z = sign * logits
                loss = z + z.exp()
            case "mse":
                sign = (2 * batch.labels - 1)
                z = sign * logits
                loss = (-2 * z).exp() - 2 * z.exp()
            case _:
                raise ValueError

        if batch.sample_weights is not None:
            loss = loss * batch.sample_weights

        if batch.w_ref_event is not None:
            loss = loss * batch.w_ref_event

        loss = loss.mean(0)
        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        return loss

    def forward(self, x: HomerData) -> torch.Tensor:
        """Return the event-level data-to-ref log-likelihood ratio"""

        if self.lowlevel:
            # mom = x.splits_for_full_hadron_info[...,9:13]
            # cond = x.hadrons_obs_only
            # logits = self.net(mom, c=cond, mask=x.accepted)
            had_mask = x.point_cloud[..., -1] != 0
            logits = self.net(x.point_cloud, mask=had_mask)
        else:
            logits = self.net(x.hadrons_obs_only)

        return logits.squeeze(1)

    def weight(self, x: HomerData) -> torch.Tensor:
        """Return the event-level data-to-ref likelihood ratio"""
        logits = self.forward(x)
        return logits.exp()

    def prob(self, x: HomerData) -> torch.Tensor:
        """Return event-level data probabilities"""
        logits = self.forward(x)
        return logits.sigmoid()
