"""
Public package exports for gmml.
"""

__all__ = [
    "GMMLayer", "DiagGMMLayer", "IsoGMMLayer",
    "get_kl_divs", "get_log_joint_probs", "get_samples",
    "get_diag_kl_divs", "get_diag_log_joint_probs", "get_diag_samples",
    "get_iso_kl_divs", "get_iso_log_joint_probs", "get_iso_samples",
]


from gmml.functional import get_kl_divs, get_log_joint_probs, get_samples
from gmml.functional import get_diag_kl_divs, get_diag_log_joint_probs, \
    get_diag_samples
from gmml.functional import get_iso_kl_divs, get_iso_log_joint_probs, \
    get_iso_samples

from gmml.layer import GMMLayer, DiagGMMLayer, IsoGMMLayer
