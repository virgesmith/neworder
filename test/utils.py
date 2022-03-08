
# test utils
from typing import Any, Callable, Type

def assert_throws(e: Type[Exception], f: Callable, *args: Any, **kwargs: Any):
  try:
    f(*args, **kwargs)
  except e:
    pass
  else:
    assert False, f"expected exception {e} not thrown by {f}"
