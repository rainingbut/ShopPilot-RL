from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("shopping-grpo")
except PackageNotFoundError:
    __version__ = "0+unknown"
