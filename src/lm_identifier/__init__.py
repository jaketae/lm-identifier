from importlib import metadata

__version__ = metadata.version(__package__)
__all__ = ["__version__", "perplexity", "position"]
