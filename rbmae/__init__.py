from .bbrbm import BBRBM
from .gbrbm import GBRBM
from .auto_encoder import AutoEncoder
from .batcher import Batcher

# default RBM
RBM = BBRBM

__all__ = [
    RBM,
    BBRBM,
    GBRBM,
    AutoEncoder,
    Batcher
]