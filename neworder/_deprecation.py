import functools
import warnings
from types import ModuleType
from typing import Any


def deprecate(module: ModuleType, name: str, replacement: str) -> None:
    """Replace a function in an extension submodule with a wrapper that emits a DeprecationWarning when called"""
    func = getattr(module, name)
    public_name = f"{module.__name__.replace('_neworder_core', 'neworder', 1)}.{name}"

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{public_name} is deprecated and will be removed in a future release. Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    setattr(module, name, wrapper)
