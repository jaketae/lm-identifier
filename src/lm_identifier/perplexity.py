from typing import List, Union

import torch
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding

from .base import rank_by


def rank_by_perplexity(
    text: str,
    model_ids: List[str],
    stride: int = 512,
    device: Union[str, torch.device] = "cpu",
):
    return rank_by(text, model_ids, get_perplexity, stride, device)


@torch.inference_mode()
def get_perplexity(
    model: PreTrainedModel,
    encodings: BatchEncoding,
    stride: int,
    device: Union[str, torch.device] = "cpu",
) -> float:
    # adapted from https://huggingface.co/docs/transformers/perplexity
    nlls = []
    prev_end_loc = 0
    max_length = model.config.max_length
    seq_len = encodings.input_ids.size(1)
    model.eval()
    model.to(device)
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    return ppl
