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
        encodings = tokenizer(text, return_tensors="pt").to(device)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        perplexity = get_perplexity(model, encodings, stride)
        result[model_name] = perplexity
    return {
        model_name: perplexity
        for model_name, perplexity in sorted(result.items(), reverse=True)
    }


def get_perplexity(
    model: PreTrainedModel, encodings: BatchEncoding, stride: int
) -> float:
    return 0
