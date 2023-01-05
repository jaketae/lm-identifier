# LM Identifier

[![PyPI](https://img.shields.io/pypi/v/lm-identifier.svg)](https://pypi.org/project/lm-identifier/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

With a surge of generative pretrained language models, it is becoming increasingly important to distinguish between human and AI-generated text. Inspired by [GPTZero](https://etedward-gptzero-main-zqgfwb.streamlit.app), an app that seeks to detect AI-generated text, LM Identifier pokes at this question even further by providing a growing suite of tools to help identify *which (publicly available) language model* might have been used to generate some given chunck of text.

## Installation

LM Identifier is available on PyPI.

```
$ pip install lm-identifier
```

To develop locally, first install pre-commit:

```
$ pip install --upgrade pip wheel
$ pip install pre-commit
$ pre-commit install
```

Install the package in editable mode.

```
$ pip install -e .
```

## Usages

**Disclaimer:** This package is under heavy development. The API and associated functionalities may undergo substantial changes.

### 1.Perplexity-based Ranking

Perplexity is a common metric used in natural language generation to measure the performance of an LM. Roughly speaking, perplexity is the exponentiation of cross entropy. The bottom line is that lower perplexity indicates higher probability of the text being generated by that model.

LM Identifier provides a perplexity-based ranking function as shown below.

```python
from lm_identifier.perplexity import rank_by_perplexity

candidate_models = [
    "gpt2",
    "distilgpt2",
    "facebook/opt-350m",
    "lvwerra/gpt2-imdb",
]

text = (
    "My name is Thomas and my main character"
    "is a young man who is a member of the military."
)

model2perplexity = rank_by_perplexity(text, candidate_models)
```

`model2perplexity` is a dictionary of perplexity scores for each language model in sorted order.

```python
{
    'lvwerra/gpt2-imdb': 13.910672187805176,
    'gpt2': 16.332365036010742,
    'facebook/opt-350m': 18.126564025878906,
    'distilgpt2': 28.430707931518555,
}
```

This toy example was indeed generated with `'lvwerra/gpt2-imdb'`, which is a standard `'gpt2'` model fine-tuned on the IMDB dataset. LM Identifier can thus be leveraged to distinguish between not only disparate models, but also an upstream model and its fine-tuned variant.

### 2. Position-based Ranking

While various autoregressive decoding and sampling methods exist, they typically involve applying a softmax over the logits to obtain the posterior distribution $p(x_t | x_1, x_2, \dots, x_{t - 1})$. We can analyze this distribution with some given text to see how closely aligned the model's predictions are with the input.

Concretely, if the token $x_t$ is ranked highly in the posterior sorted by probability mass, it is likely to have been produced by the model.

```python
>>> from lm_identifier.position import rank_by_position
>>> model2position = rank_by_position(text, candidate_models)
>>> model2position
{
    'lvwerra/gpt2-imdb': 38.94736842105263,
    'gpt2': 50.78947368421053,
    'facebook/opt-350m': 100.6842105263158,
    'distilgpt2': 318.89473684210526
}
```

On average, the tokens that appear in the input text ranked 40 in `'lvwerra/gpt2-imdb'`. While the ranking score may not seen low enough, recall that GPT-2 has an token vocabulary size of 50257. Given the cardinality of the PMF domain, this is a low score. Note also that the result is aligned with that obtained through ranking by perplexity.

## Acknowledgement

This project heavily borrows code from Hugging Face's article on [perplexity measurement](https://huggingface.co/docs/transformers/perplexity), as well as the [GLTR code base](https://github.com/HendrikStrobelt/detecting-fake-text).

This project was heavily inspired by [GPTZero](https://etedward-gptzero-main-zqgfwb.streamlit.app), a project by [Edward Tian](https://twitter.com/edward_the6/status/1610067688449007618).

## License

Released under the [MIT License](License).
