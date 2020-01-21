""" greet.py """

import os
import neworder # to pass greeting back to environment

class Greet():
  # Constructor 
  def __init__(self, *args):
    self.name = "anonymous"
    self.hellos = args

  # Gets username
  def get_name(self):
    self.name = os.getlogin()

  def __call__(self):
    for h in self.hellos:
      neworder.log(h + " " + self.name)

    #neworder.log(dir(neworder))
