"""EOB models. Each module needs its own waveform package, so they are imported on demand:
`get_model("SEOBNRv5PHM")` returns the class."""

from importlib import import_module

MODELS = {
    "SEOBNRv5PHM": ("seobnrv5", "SEOBNRv5PHM"),
    "SEOBNRv4PHM": ("seobnrv4", "SEOBNRv4PHM"),
    "TEOBResumS-Dali": ("teob_dali", "TEOBResumSDali"),
}


def get_model(approximant):
    module, cls = MODELS[approximant]
    return getattr(import_module(f"{__name__}.{module}"), cls)
