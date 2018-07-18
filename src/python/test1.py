

#class Person:

import collections

Person = collections.namedtuple('Person', ['id', 'location', 'age', 'gender', 'ethnicity'])
#Person.__new__.__defaults__ = (0, "", 0, "M", "UNK")

def run():

  print('[python] Type of Person:', type(Person))

  bob = Person(id=0, age=30, gender='male', location="E09000001", ethnicity=1)
  print('\n[python] Representation:', bob)

  jane = Person(id=1, age=29, gender='female', location="", ethnicity="BLA")
  print('\n[python] Field by name:', jane.location)
  print('\n[python] Fields by index:', bob[1])
  for p in [ bob, jane ]:
      print("[python]", p)
  return bob
  