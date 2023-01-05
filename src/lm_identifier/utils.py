from numbers import Number
from typing import Dict


def sort_by_value(dictionary: Dict[str, Number], **kwargs):
    return dict(sorted(dictionary.items(), key=lambda x: x[1], **kwargs))
