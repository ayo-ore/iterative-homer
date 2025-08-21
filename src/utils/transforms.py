import torch
from tensordict import tensorclass

from .utils import get_break_masks


class ShiftAndScale:
    """
    x -> (x - shift) / scale
    """

    @staticmethod
    def forward(x, shift=0, scale=1):
        return x.sub(shift).div(scale)

    @staticmethod
    def reverse(x, shift=0, scale=1):
        return x.multiply(scale).add(shift)


@tensorclass
class HadronObsTransform:
    """
    Transform hadron observables by standardizing to zero mean and unit variance
    """

    shift: torch.Tensor
    scale: torch.Tensor

    def forward(self, x):

        # set NaNs to zero
        x.observables = x.observables.nan_to_num()

        # normalize
        x.observables = ShiftAndScale.forward(
            x.observables, shift=self.shift, scale=self.scale
        )

        return x

    def reverse(self, x):

        # unnormalize
        x.observables = ShiftAndScale.reverse(
            x.observables, shift=self.shift, scale=self.scale
        )
        return x


class HadronMassTransform:
    """ """  # TODO

    masses = [0.13498, 0.13957]

    def forward(self, x):
        y = x
        for i, m in enumerate(self.masses):
            y = torch.where(x.isclose(torch.tensor(m)), i, y)
        return y

    def reverse(self, x):
        y = x
        for i, m in enumerate(self.masses):
            y = torch.where(x == i, m, y)
        return y


@tensorclass
class HistoryTransform:
    """
    TODO
    """

    keep_stringends: bool
    shift: torch.Tensor
    scale: torch.Tensor
    eps: float = 1e-6
    mass_transform: HadronMassTransform = HadronMassTransform()
    permutation: torch.Tensor = torch.tensor([4, 3, 1, 2, 0, 5, 6, 7, 8, 9, 10, 11, 12])

    def forward(self, x):

        # y = x.breaks.clone()

        # add masks
        x.is_break, x.accepted, _ = get_break_masks(
            x, keep_stringends=self.keep_stringends
        )

        # convert masses to classes
        x.breaks[..., 3] = self.mass_transform.forward(x.breaks[..., 3])

        # logit transform momentum fractions
        x.breaks[..., 0] = torch.logit(x.breaks[..., 0] * (1 - 2 * self.eps) + self.eps)

        # normalize
        x.breaks = ShiftAndScale.forward(x.breaks, shift=self.shift, scale=self.scale)

        # permute feature order: frompos, mass, delta_pT, z, pT_string
        x.breaks = x.breaks[..., self.permutation]

        return x

    def reverse(self, x):

        # y = x.breaks.clone()

        # create accepted mask
        x.accepted = (x.breaks != 0).any(-1)

        # undo permutation
        x.breaks = x.breaks[..., self.permutation.argsort()]

        # unnormalize
        x.breaks = ShiftAndScale.reverse(x.breaks, shift=self.shift, scale=self.scale)

        # restore momentum fractions
        x.breaks[..., 0] = (x.breaks[..., 0].sigmoid() - self.eps) / (1 - 2 * self.eps)

        # restore masses from classes
        x.breaks[..., 3] = self.mass_transform.reverse(x.breaks[..., 3])

        # re-enforce mask
        x.breaks[~x.accepted] = 0.0

        # x.breaks = y

        return x


@tensorclass
class PointCloudTransform:
    """
    TODO
    """

    shift: torch.Tensor
    scale: torch.Tensor

    def forward(self, x):

        # log-scale energy
        x.point_cloud[..., 3] = x.point_cloud[..., 3].add(1e-6).log()

        # normalize
        x.point_cloud = ShiftAndScale.forward(
            x.point_cloud, shift=self.shift, scale=self.scale
        )

        return x

    def reverse(self, x):

        # unnormalize
        x.point_cloud = ShiftAndScale.reverse(
            x.point_cloud, shift=self.shift, scale=self.scale
        )

        # exp scale energy
        x.point_cloud[..., 3] = x.point_cloud[..., 3].exp()

        return x
