""" greet.py """

import os
import neworder # to pass greeting back to environment

class Greet():
  # Constructor 
  def __init__(self, *args): #, **kwargs):
    self.name = "anonymous"
    self.hellos = args
    #self.model = HelloWorld(neworder.Timeline(2.7183, 3.1416, [10]))

  # Gets username
  def set_name(self):
    self.name = os.getlogin()

  def __call__(self):
    for h in self.hellos:
      neworder.log("%s %s" % (h, self.name))
    #return 2.718281828

    #neworder.log(dir(neworder))


