# LM Identifier

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

```python
from lm_identifier import rank

candidate_models = [
    "gpt2",
    "distilgpt2",
    "facebook/opt-350m",
    "lvwerra/gpt2-imdb",
]

text = "My name is Thomas and my main character is a young man who is a member of the military."

model2perplexity = rank(text, candidate_models)
```

`model2perplexity` is a dictionary of perplexity scores for each language model, sorted in descending order.

## Acknowledgement

This project heavily borrows code from Hugging Face's article on [perplexity measurement](https://huggingface.co/docs/transformers/perplexity).

This project was heavily inspired by [GPTZero](https://etedward-gptzero-main-zqgfwb.streamlit.app), a project by [Edward Tian](https://twitter.com/edward_the6/status/1610067688449007618).

## License

Released under the [MIT License](License).
