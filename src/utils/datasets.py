import numpy as np
import os
import torch

from tensordict import tensorclass

from typing import List, Optional

KEYS = (
    "analytical_log_weight",
    "hadrons_obs_only",
    "history_indexes",
    "splits_for_full_hadron_info",
    "point_cloud",
    "analytical_per_split_log_weight",
    "sample_weights",
)


@tensorclass
class HomerData:
    """
    A tensorclass holding hadronization data for the HOMER method.
    The optional attributes are
        - "analytical_log_weight"           (N_events,):                  TODO: Description
        - "hadrons_obs_only"                (N_events, 13):               TODO: Description
        - "history_indexes"                 (N_events, 100):              TODO: Description
        - "splits_for_full_hadron_info"     (N_events, 100, 13):          TODO: Description
        - "point_cloud"                     (N_events, 60, 5):            TODO: Description
        - "analytical_per_split_log_weight" (N_events, 100):              TODO: Description
    The following attributes are placeholders, filled internally by HOMER.
        - "labels"                          (N_events, 1):                TODO: Description
        - "is_break"                        (N_events, 100):              TODO: Description
        - "accepted"                        (N_events, 100):              TODO: Description
        - "num_rej"                         (N_events,):                  TODO: Description
        - "in_chain_n"                      (N_events, 100, max_num_rej)  TODO: Description
        - "w_class"                         (N_events,):                  TODO: Description
        - "w_ref_break"                     (N_events, 100):              TODO: Description
        - "w_ref_chain"                     (N_events,):                  TODO: Description
        - "w_ref_history"                   (N_events,):                  TODO: Description
        - "w_ref_event"                     (N_events,):                  TODO: Description
        - "sample_weights"                  (N_events,):                  TODO: Description
    """

    # fmt: off
    analytical_log_weight:           Optional[torch.Tensor] = None
    hadrons_obs_only:                Optional[torch.Tensor] = None
    history_indexes:                 Optional[torch.Tensor] = None
    splits_for_full_hadron_info:     Optional[torch.Tensor] = None
    point_cloud:                     Optional[torch.Tensor] = None
    analytical_per_split_log_weight: Optional[torch.Tensor] = None
    labels:                          Optional[torch.Tensor] = None
    is_break:                        Optional[torch.Tensor] = None
    accepted:                        Optional[torch.Tensor] = None
    num_rej:                         Optional[torch.Tensor] = None
    in_chain_n:                      Optional[torch.Tensor] = None
    w_class:                         Optional[torch.Tensor] = None
    w_ref_break:                     Optional[torch.Tensor] = None
    w_ref_event:                     Optional[torch.Tensor] = None
    sample_weights:                  Optional[torch.Tensor] = None
    # fmt: on

    @classmethod
    def from_dir(
        cls,
        path: str,
        num: int = -1,
        w_class_path: str = None,
        w_prev_path: str = None,
        keys: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):

        # check keys
        keys = resolve_keys(keys)
        
        # always inlcude sample weights
        keys.append("sample_weights")

        # read tensors into memory
        tensor_kwargs = {}
        for k in keys:

            try:
                filename = os.path.join(path, k + ".npy")
                array = np.load(filename)[:num]
                tensor_kwargs[k] = torch.from_numpy(array)
            
            except FileNotFoundError as e:
            
                if k == "sample_weights":
                    tensor_kwargs["sample_weights"] = torch.ones(len(array))
                else:
                    raise e

        # fill placeholder fields with results from previous runs
        filename = "predictions" + ("" if path.endswith("test") else "_train")
        is_data = os.path.basename(path).startswith("data")

        if w_class_path is not None:  # load step one classifier weights

            record = np.load(os.path.join(w_class_path, f"{filename}.npz"))
            w_class = record["weights"]
            w_class = w_class[..., record["labels"] == int(is_data)]
            tensor_kwargs["w_class"] = torch.from_numpy(w_class).mean(0)

        if w_prev_path is not None:  # load previous iteration weights

            record = np.load(os.path.join(w_prev_path, f"{filename}.npz"))
            if is_data:
                w_ref_break = torch.ones((len(array), 100))
                w_ref_event = torch.ones(len(array))
            else:
                w_ref_break = torch.from_numpy(record["break_weights"]).mean(0)
                w_ref_event = torch.from_numpy(record["event_weights"]).mean(0)

                # fix normalization (only affects training, not final output)
                w_ref_event /= w_ref_event.mean()

            tensor_kwargs["w_ref_break"] = w_ref_break.nan_to_num(1.0)
            tensor_kwargs["w_ref_event"] = w_ref_event

        # return tensorclass dataset
        return cls(batch_size=[len(array)], device=device, **tensor_kwargs)


def resolve_keys(keys):

    # use all keys by default
    keys = keys or KEYS

    # check that all keys are known
    unknown_keys = set(keys) - set(KEYS)
    assert unknown_keys == set(), f"Found unknown keys {unknown_keys}"

    return keys
