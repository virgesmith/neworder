""" people.py """

from enum import Enum
import pandas as pd
import neworder


class Gender(Enum):
  MALE = True
  FEMALE = False

def is_male(gender):
  return gender

def is_female(gender):
  return not gender

def active_age(age):
  return (age >= 15) and (age < retirement_age(age))

class RelationshipStatus(Enum):
  SINGLE = 1
  MARRIED = 2
  DIVORCED = 3
  WIDOW = 4

def is_single(status):
  return status == RelationshipStatus.SINGLE
def is_married(status):
  return status == RelationshipStatus.MARRIED
def is_divorced(status):
  return status == RelationshipStatus.DIVORCED
def is_widow(status):
  return status == RelationshipStatus.WIDOW

# globals:
#     periodic:
#         - RETIREMENTAGE: int
class RetirementAge():
  def __init__(self, input_data, key):
    # store in array as dataframe lookup is less efficient
    retirement_table = pd.read_hdf(input_data, key)
    self.offset = retirement_table.PERIOD.values[0]
    self.lookup = retirement_table.RETIREMENTAGE.values

  def __call__(self, year):
    return self.lookup[year - self.offset]

class People():

  #  macros:
  UNSET = -1

  def __init__(self, input_data, key):
    self.pp = pd.read_hdf(input_data, key)

    # fields:
    # ['period' 'id' 'age' 'gender' 'workstate' 'civilstate' 'dur_in_couple' 'mother_id' 'partner_id' 'hh_id']
    #     # period and id are implicit
    #     - age:        int
    #     - gender:     bool
    #     # 1: single, 2: married, 3: divorced, 4: widowed
    #     - civilstate: int
    #     - dur_in_couple: int

    #     # link fields
    #     - mother_id:  int
    #     - partner_id: int
    #     - hh_id:      int

    #     # fields not present in input
    #     - agegroup_civilstate: {type: int, initialdata: False}

    # links:
    #     mother: {type: many2one, target: person, field: mother_id}
    #     partner: {type: many2one, target: person, field: partner_id}
    #     household: {type: many2one, target: household, field: hh_id}
    #     children: {type: one2many, target: person, field: mother_id}


    # possible transitions and regressions
    def ageing(self):
      pass
    # - age: age + 1
    # - agegroup_civilstate: if(age < 50,
    #                           5 * trunc(age / 5),
    #                           10 * trunc(age / 10))

    def birth(self):
      pass
    # - to_give_birth: logit_regr(0.0,
    #                             filter=ISFEMALE and (
    #                                 age >= 15) and (age <= 50),
    #                             align='al_p_birth.csv')
    # - new('person', filter=to_give_birth,
    #       mother_id=id,
    #       hh_id=hh_id,
    #       partner_id=UNSET,
    #       civilstate=SINGLE,
    #       age=0,
    #       agegroup_civilstate=0,
    #       gender=choice([MALE, FEMALE], [0.51, 0.49]))

    def death(self):
      pass
        # - dead: if(ISMALE,
        #             logit_regr(0.0, align='al_p_dead_m.csv'),
        #             logit_regr(0.0, align='al_p_dead_f.csv'))
        # - mother_id: if(mother.dead, UNSET, mother_id)
        # - civilstate: if(partner.dead, WIDOW, civilstate)
        # - partner_id: if(partner.dead, UNSET, partner_id)
        # - show('Avg age of dead men', avg(age, filter=dead and ISMALE))
        # - show('Avg age of dead women', avg(age, filter=dead and ISFEMALE))
        # - show('Widows', count(ISWIDOW))
        # - remove(dead)

    def marriage(self):
      pass
#         - to_couple: if((age >= 18) and (age <= 90) and not ISMARRIED,
#                         if(ISMALE,
#                             logit_regr(0.0, align='al_p_mmkt_m.csv'),
#                             logit_regr(0.0, align='al_p_mmkt_f.csv')),
#                         False)
#         # abs() simply returns the absolute value of its argument
#         # this ranks women depending on their distance to the average
#         # age of available men
#         - difficult_match: if(to_couple and ISFEMALE,
#                               abs(age - avg(age, filter=to_couple and ISMALE)),
#                               nan)
#         - partner_id: if(to_couple,
#                           matching(set1filter=ISFEMALE, set2filter=ISMALE,
#                                   score=- 0.4893 * other.age
#                                   + 0.0131 * other.age ** 2
#                                   - 0.0001 * other.age ** 3
#                                   + 0.0467 * (other.age - age)
#                                   - 0.0189 * (other.age - age) ** 2
#                                   + 0.0003 * (other.age - age) ** 3,
#                                   orderby=difficult_match),
#                           partner_id)
#         - justcoupled: to_couple and (partner_id != UNSET)

#         # When using links, one is not limited to retrieve simple fields
#         # but can compute expressions directly in the linked
#         # individuals. To do that use the .get method on many2one links.
#         - alone: household.get(persons.count() == 1)

#         # create new households for newly wed women
#         - needhousehold: ISFEMALE and justcoupled and not alone
#         - in_mother_hh: hh_id == mother.hh_id
#         - hh_id: if(needhousehold, new('household'), hh_id)
#         # bring along their husband
#         - hh_id: if(ISMALE and justcoupled, partner.hh_id, hh_id)
#         # bring along their children, if any
#         - hh_id: if(mother.justcoupled and in_mother_hh,
#                     mother.hh_id,
#                     hh_id)

#         - civilstate: if(justcoupled, MARRIED, civilstate)
#         - dur_in_couple: if(justcoupled,
#                             0,
#                             if(ISMARRIED, dur_in_couple + 1, 0))
# #                - csv(dump(id, age, gender, partner.id, partner.age,
# #                           partner.gender, hh_id, filter=justcoupled),
# #                      suffix='new_couples')

    def get_a_life(self):
      pass
        # # create new households for persons aged 24+ who are still
        # # living with their mother
        # - in_mother_hh: hh_id == mother.hh_id
        # - should_move: in_mother_hh and (age >= 24)
        # - hh_id: if(should_move, new('household'), hh_id)
        # # bring along their children, if any
        # - hh_id: if(in_mother_hh and mother.should_move,
        #             mother.hh_id,
        #             hh_id)

    def divorce(self):
      pass
        # - agediff: if(ISFEMALE and ISMARRIED, age - partner.age, 0)
        # - nb_children: household.get(persons.count(age < 18))
        # # select females to divorce

        # # here we use logit_regr with an actual expression, which means
        # # that within each group, persons with a high value for the
        # # given expression will be selected first.
        # - divorce: logit_regr(0.6713593 * nb_children
        #                       - 0.0785202 * dur_in_couple
        #                       + 0.1429621 * agediff - 0.0088308 * agediff ** 2
        #                       - 4.546278,
        #                       filter=ISFEMALE and ISMARRIED and (
        #                           dur_in_couple > 0),
        #                       align='al_p_divorce.csv')
        # # break link to partner
        # - to_divorce: divorce or partner.divorce
        # - partner_id: if(to_divorce, UNSET, partner_id)

        # - civilstate: if(to_divorce, DIVORCED, civilstate)
        # - dur_in_couple: if(to_divorce, 0, dur_in_couple)
        # # move out males
        # - hh_id: if(ISMALE and to_divorce,
        #             new('household'),
        #             hh_id)

    def civilstate_changes(self):
      pass
        # the lag() function returns the value of any expression
        # at the end of the previous period. Note that for non-aggregate
        # expressions, it returns a value for all individuals present
        # in the *current* period. For individuals which were not
        # present (born) in the previous period, it returns the
        # "missing" value (-1 for int fields, like in this case).
        #- show(groupby(civilstate, lag(civilstate)))

    def chart_demography(self):
      pass
        # - bar(groupby(agegroup_civilstate, gender),
        #       fname='demography_{period}.png')

    def csv_output(self):
      pass
        # - csv(groupby(agegroup_civilstate, gender),
        #       fname='demography_{period}.csv')

        # - csv(period,
        #       avg(age, filter=ISMALE), avg(age, filter=ISFEMALE),
        #       avg(age),
        #       fname='average_age.csv', mode='a')
        # - csv(period, count(not ACTIVEAGE) / count(ACTIVEAGE),
        #       fname='dependency_ratio.csv', mode='a')

    def init_reports(self):
      pass
        # - csv('period', 'men', 'women', 'total',
        #       fname='average_age.csv')
        # - csv('period', 'ratio', fname='dependency_ratio.csv')
