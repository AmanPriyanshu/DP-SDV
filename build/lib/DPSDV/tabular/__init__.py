"""Models for tabular data."""

from DPSDV.tabular.copulagan import CopulaGAN
from DPSDV.tabular.copulas import GaussianCopula
from DPSDV.tabular.ctgan import CTGAN, TVAE
from DPSDV.tabular.mwem import MWEMSynthesizer

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'TVAE',
    'CopulaGAN',
    'MWEMSynthesizer',
)
