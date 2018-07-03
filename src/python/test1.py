

#class Person:

import collections

Person = collections.namedtuple('Person', ['id', 'location', 'age', 'gender', 'ethnicity'])
#Person.__new__.__defaults__ = (0, "", 0, "M", "UNK")

def run():

  print('Type of Person:', type(Person))

  bob = Person(id=0, age=30, gender='male', location="E09000001", ethnicity=1)
  print('\nRepresentation:', bob)

  jane = Person(id=1, age=29, gender='female', location="", ethnicity="BLA")
  print('\nField by name:', jane.location)

  print('\nFields by index:')
  for p in [ bob, jane ]:
      print(p)
  return 5
  