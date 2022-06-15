"""Models for tabular data."""

from DPSDV.tabular.copulagan import CopulaGAN
from DPSDV.tabular.copulas import GaussianCopula
from DPSDV.tabular.ctgan import CTGAN, TVAE

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'TVAE',
    'CopulaGAN',
)
