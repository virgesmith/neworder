""" greet.py """

import os
import neworder # to pass greeting back to environment

class Greet():
  # Constructor 
  def __init__(self):
    self.name = "anonymous"

  # Gets username
  def get_name(self):
    self.name = os.getlogin()

  def __call__(self):
    neworder.log("Hello " + self.name)
