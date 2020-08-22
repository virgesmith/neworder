
# this test must be run first 
import neworder as no

# ensure initialised (remains so for the duration of pytest) 
no.module_init(independent=False, verbose=False)

def test_basics():
  # just check you can call the functions
  no.name()
  no.version()
  no.python()
  assert no.mpi.indep() == False
  assert no.verbose() == False
  no.log("testing")
  assert not no.embedded()

  try:
    no.module_init()
  except Exception as e:
    pass
  else:
    assert False, "expected exception on second initialisation not thrown"
