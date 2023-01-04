from typing import List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding


def rank(
    text: str,
    model_names: List[str],
    stride: int = 512,
    device: Union[str, torch.device] = "cpu",
):
    # TODO: parallelize
    # TODO: support batches
    result = {}
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(text, return_tensors="pt")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        perplexity = get_perplexity(model, encodings, stride, device)
        result[model_name] = perplexity
    return {
        model_name: perplexity
        for model_name, perplexity in sorted(result.items(), reverse=True)
    }


@torch.inference_mode()
def get_perplexity(
    model: PreTrainedModel,
    encodings: BatchEncoding,
    stride: int,
    device: Union[str, torch.device] = "cpu",
) -> float:
    nlls = []
    prev_end_loc = 0
    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)
    model.eval()
    model.to(device)
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl
