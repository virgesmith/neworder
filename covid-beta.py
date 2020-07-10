#%%

import pandas as pd
import datetime

base_url = "https://api.coronavirus-staging.data.gov.uk/v1/data"

ltla_url = base_url + "?filters=areaType=ltla&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22date%22:%22date%22,%22newCasesBySpecimenDate%22:%22newCasesBySpecimenDate%22,%22cumCasesBySpecimenDate%22:%22cumCasesBySpecimenDate%22%7D&format=csv"

wales = ['Isle of Anglesey', 'Gwynedd',
 'Conwy', 'Denbighshire', 'Flintshire', 'Wrexham', 'Ceredigion',
 'Pembrokeshire', 'Carmarthenshire', 'Swansea', 'Neath Port Talbot', 'Bridgend', 'Vale of Glamorgan', 'Cardiff', 'Rhondda Cynon Taf', 'Caerphilly', 'Blaenau Gwent', 'Torfaen', 'Monmouthshire', 'Newport', 'Powys',
 'Merthyr Tydfil']

ltla = pd.read_csv(ltla_url)
ltla["date"] = pd.to_datetime(ltla["date"])
ltla.sort_values("date", inplace=True)

ltla = ltla[~ltla["areaName"].isin(wales)].reset_index(drop=True)
print(ltla.head())
timestamp = datetime.datetime.now()
filename = "./covid_cases_%s.csv" % timestamp.strftime("%Y-%m-%d")
ltla.to_csv(filename)
print("Saved data to %s" % filename)
# should be 315
assert len(ltla["areaName"].unique()) == 315

#eng = pd.read_csv(eng_url)
# %%

places = ["Bradford", "Leeds", "Leicester"]
#places = ["Bradford", "Leeds", "Leicester"]
mye = pd.read_csv("./mye2019.csv")

pops = mye[mye["Name"].isin(["England"] + places)][["Name", "All ages"]].set_index("Name").to_dict()["All ages"]
print("Populations (2019MYE):", pops)

#%%

eng = ltla.groupby(["date"]).sum().reset_index() #["newCasesBySpecimenDate"].sum()

eng['newCasesBySpecimenDate'] = eng['newCasesBySpecimenDate'] / pops["England"] * 100000.0
eng["newCasesWeeklyMean"] = eng['newCasesBySpecimenDate'].rolling(window=7).mean()
print(eng.head())


#%%

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cols = ['date', 'newCasesBySpecimenDate']

# first add england
plt.plot(eng["date"], eng['newCasesWeeklyMean'])

for place in places:
  lad = ltla[ltla["areaName"] == place][cols]
  lad = lad.sort_values(by="date").reset_index(drop=True)
  lad['newCasesBySpecimenDate'] = lad['newCasesBySpecimenDate'] / pops[place] * 100000.0
  lad["newCasesWeeklyMean"] = lad['newCasesBySpecimenDate'].rolling(window=7).mean()

  print(lad.tail(10))

  plt.plot(lad["date"], lad['newCasesWeeklyMean'])

plt.xlabel("Specimen date")  
plt.ylabel("New cases per 100,000 people")  
plt.legend(["England"] + places)
plt.title("New case rates, 7 day moving average\nData from %s sourced on %s" % (base_url, timestamp.strftime("%Y-%m-%d %H:%M:%S")))
fig = plt.gcf()
fig.set_size_inches(16.0, 10.0)


# %%

eng_url = base_url + "?filters=areaType=nation;areaName=England&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22date%22:%22date%22,%22plannedCapacityByPublishDate%22:%22plannedCapacityByPublishDate%22,%22newTestsByPublishDate%22:%22newTestsByPublishDate%22,%22cumTestsByPublishDate%22:%22cumTestsByPublishDate%22%7D&format=csv"

tests = pd.read_csv(eng_url)
tests["date"] = pd.to_datetime(tests["date"])
tests.sort_values("date", inplace=True)
#ltla = ltla[~ltla["areaName"].isin(wales)].reset_index(drop=True)
print(tests.head())

# %%
plt.plot(tests["date"], tests["newTestsByPublishDate"])


# %%
