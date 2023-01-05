from importlib import metadata

from lm_identifier.core import rank

__version__ = metadata.version(__package__)
__all__ = ["rank", "__version__"]
