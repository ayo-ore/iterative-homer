import torch
from hydra.utils import instantiate

from .datasets import HomerData
from .config import get_prev_config


def load_model(path, model_cls=None, device=None):

    cfg = get_prev_config(path)

    # TODO: tidy this to just use instantiate
    model = model_cls(cfg) if model_cls is not None else instantiate(cfg.model, cfg=cfg)
    # model = instantiate(cfg.model, cfg=cfg)
    sdict = torch.load(path + "/model.pt", weights_only=False)
    model.load_state_dict(sdict["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model = model.to(device)

    # model = torch.compile(model)

    return model, cfg


def get_break_masks(x: HomerData, keep_stringends=False):
    # TODO: Document and change arguments to I, H

    I = x.history_indexes

    # non-padded entries
    is_break = I != -1

    # breaks in accepted chain
    accepted = I == I.max(dim=1, keepdim=True).values

    # mask indicating if break is in chain n (new dimension at pos 0)
    num_rej = I.max(1).values  # number of rejected chains per event
    chain_index = torch.arange(num_rej.max() + 1, device=x.device)
    in_chain_n = I == chain_index.view(-1, 1, 1)

    # Optionally remove string ends from masks
    if not keep_stringends:
        is_stringend = x.splits_for_full_hadron_info[..., 8] == 1
        is_break &= ~is_stringend
        accepted &= ~is_stringend
        in_chain_n &= ~is_stringend
        x.history_indexes[is_stringend] = -1

    return is_break, accepted, in_chain_n


def get_last_break_mask(history_indices: torch.Tensor, n=1):
    """
    Constructs a mask selecting the nth-last string breaks per chain.

    Parameters:
    history_indices (torch.Tensor): Tensor of shape (N_samples, 100) containing chain
                                    indices of each break.

    Returns:
    mask (torch.Tensor): A boolean mask of shape (N_samples, 100)
    """

    I = history_indices

    mask = torch.ones_like(I, dtype=bool)
    mask &= I != -1  # ignore padded entries
    mask[:, :-n] &= I[:, n:] != I[:, :-n]  # select final n breaks per chain
    mask[:, 1:] &= mask[:, :-1] ^ mask[:, 1:]  # select nth last break
    return mask
