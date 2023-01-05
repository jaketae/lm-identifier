from typing import Callable, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import sort_by_value


def rank_by(
    text: str,
    model_ids: List[str],
    score_fn: Callable,
    *args,
    **kwargs,
):
    # TODO: parallelize
    # TODO: support batches
    model2score = {}
    for model_id in model_ids:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encodings = tokenizer(text, return_tensors="pt")
        score = score_fn(model, encodings, *args, **kwargs)
        model2score[model_id] = score
    return sort_by_value(model2score)
