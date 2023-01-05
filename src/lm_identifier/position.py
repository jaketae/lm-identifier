from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from .base import rank_by


def rank_by_position(
    text: str,
    model_ids: List[str],
    stride: int = 512,
    device: Union[str, torch.device] = "cpu",
):
    return rank_by(text, model_ids, get_predicted_position, stride, device)


@torch.inference_mode()
def get_predicted_position(
    model: PreTrainedModel,
    encodings: BatchEncoding,
    stride: int,
    device: Union[str, torch.device] = "cpu",
) -> float:
    # from https://github.com/HendrikStrobelt/detecting-fake-text
    positions = []
    max_length = model.config.max_length
    seq_len = encodings.input_ids.size(1)
    model.eval()
    model.to(device)
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        y = input_ids.squeeze()[1:]
        outputs = model(input_ids)
        all_logits = outputs.logits.squeeze()[:-1]
        all_probs = torch.softmax(all_logits, dim=1)
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        real_topk_pos = [
            int(np.where(sorted_preds[i] == y[i].item())[0][0])
            for i in range(y.shape[0])
        ]
        positions.append(np.mean(real_topk_pos))
        if end_loc == seq_len:
            break
    return np.mean(positions)
