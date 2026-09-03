"""
Package for creating e-graphs in Python.
"""

from . import config, ipython_magic  # noqa: F401
from .bindings import (  # noqa: F401
    EggSmolError,
    StageInfo,
    TimeOnly,
    WithPlan,
    fingerprint_ops_64,
    fingerprint_ops_128,
)
from .builtins import *  # noqa: UP029
from .conversion import *
from .deconstruct import *
from .egraph import *
from .egraph import ActionLike as ActionLike
from .runtime import define_expr_method as define_expr_method

del ipython_magic
