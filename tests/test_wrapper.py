import unittest
import os

class Test(unittest.TestCase):

  def test_serial(self):
    res = os.system("make test")
    self.assertTrue(res == 0)

  def test_mpi(self):
    res = os.system("make -f MPI.mk test")
    self.assertTrue(res == 0)

