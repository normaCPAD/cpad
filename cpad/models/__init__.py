from cpad.models.base import CPADModel
from cpad.models.discrete import DiscreteCPAD
from cpad.models.linear import LinearCPAD
from cpad.models.order import OrderCPAD
from cpad.models.ensemble import EnsembleCPAD
from cpad.models.routed import RoutedCPAD

__all__ = ["CPADModel", "DiscreteCPAD", "LinearCPAD", "OrderCPAD",
           "EnsembleCPAD", "RoutedCPAD"]


def GatedCPAD(*args, **kwargs):
    """Lazy accessor for the torch-based variant (keeps torch an optional dependency)."""
    from cpad.models.gated import GatedCPAD as _GatedCPAD
    return _GatedCPAD(*args, **kwargs)
