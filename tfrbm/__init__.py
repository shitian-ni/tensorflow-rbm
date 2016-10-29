import pkg_resources

from .bbrbm import BBRBM
from .gbrbm import GBRBM

# default RBM
RBM = BBRBM

__all__ = [RBM, BBRBM, GBRBM]
__version__ = pkg_resources.require('tfrbm')[0].version
