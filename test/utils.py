
# test utils

def assert_throws(e, f, *args, **kwargs):
  try:
    f(*args, **kwargs)
  except e:
    pass
  else:
    assert False, "expected exception %s not thrown" % e
