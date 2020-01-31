""" greet.py """

import os
import neworder # to pass greeting back to environment

class Greet():
  # Constructor 
  def __init__(self, *args): #, **kwargs):
    self.name = "anonymous"
    self.hellos = args

  # Gets username
  def get_name(self):
    self.name = os.getlogin()
    #return self.name

  def __call__(self):
    for h in self.hellos:
      neworder.log(h + " " + self.name)
    return 2.718281828

    #neworder.log(dir(neworder))
