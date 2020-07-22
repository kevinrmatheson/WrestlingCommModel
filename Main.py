import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\Kevin\Downloads\MASTER_COMM_LIST.csv") #Just make sure using most recent file, make sure to change your  path
data = data.iloc[:,[0,1,2,3,4,6]]
### Make sure to reorder the data by LAST NAME###
### CHANGE THE CURRENT DATE HERE ###
year = 2020
mo = 6
day = 22
today = (year - 1) * 365 + (mo-1) * 30 + (day)
columns = ['Wrestler', 'Grad_Year', 'Contact_Type', 'Contact_By', 'Date', 'Start_Time']
data.columns = columns
days = []
for row, comm in data.iterrows():
  text = comm.Date
  mo1 = re.findall(r'\d+', text)[0]
  day1 = re.findall(r'\d+', text)[1]
  year1 = re.findall(r'\d+', text)[2]
  days.append((int(year1) - 1) * 365 + (int(mo1)-1) * 30 + (int(day1)))

###Creating SMS interaction Model###
#First count all SMS by date
#Next note all dates where >= 2 SMSs occured -> CoachSMS
#Next for each student count num of times >=1 SMS s.t. Date is 0-3 days ahead of any CoachSMS -> StuSMS
data['DateTime'] = [str(data.Date[row]) +" "+ str(data['Start_Time'][row]) for row in range(data.shape[0])]
SMSCounts = []
TimeLists = []
DateCounter = Counter(data[data['Contact_Type'] == 'SMS'].loc[:,'DateTime'])
MinSMS = 2 #minimum number of SMSs at the exact same time to classify as bulk
data['Last_Comm'] = [today - day for day in days]#approx last communcation
data['Last_Open'] = [(data['Last_Comm'][item] if data['Contact_Type'][item] == 'Email Opened' else 9999) for item in range(data.shape[0])]#perhaps use since not EMail_Not_Opened?
S_Comm = [0] * data.shape[0]
C_Comm = [0] * data.shape[0]
###SMSTally tags SMSs as being sent by students IF the date of sms is not part of a bulk list as defined in MinSMS
data['SMSTally'] = [(1 if data['Contact_Type'][item] == 'SMS' and DateCounter[data['DateTime'][item]] <= MinSMS else 0) for item in range(data.shape[0])]
recent = 30 #this will determe how recent recent comms are defined to be
recent_data = data[data['Last_Comm'] < recent]
#Comms = Counter(data.loc[:,'Contact_Type']).keys()
#Comm_Nums = Counter(data.loc[:,'Contact_Type']).values()

names = data.Wrestler.unique()
Communications = []#dictionary of different communication types and amounts
rec_comms = []
#Comm_Amount = []

for name in names:
  Communications.append(Counter(data[data.Wrestler == name].loc[:,'Contact_Type']))
  rec_comms.append(Counter(recent_data[recent_data.Wrestler == name].loc[:, 'Contact_Type']))
  #Comm_Amount.append(sum(list(Counter(data[data.Wrestler == name].loc[:,'Contact_Type']).values())))

people = len(names)

comm_dict = Counter(data.loc[:,'Contact_Type'])

opens = [Communications[row]['Email Opened'] for row in range(people)]
nopens = [Communications[row]['Email Not Opened'] for row in range(people)]
smss = [Communications[row]['SMS'] for row in range(people)]
outcall = [Communications[row]['Outgoing Call'] for row in range(people)]
inmail = [Communications[row]['Incoming Email'] for row in range(people)]
outmail = [Communications[row]['Outgoing Email'] for row in range(people)]
incall = [Communications[row]['Incoming Call'] for row in range(people)]
visits = [Communications[row]['Official On-Campus Visit'] for row in range(people)]
general = [Communications[row]['General'] for row in range(people)]
voice = [Communications[row]['Voice Mail'] for row in range(people)]
offcon = [Communications[row]['Off-Campus Contact'] for row in range(people)]
eval = [Communications[row]['Evaluation'] for row in range(people)]
post = [Communications[row]['Postal Mail'] for row in range(people)]
###If a new type of comm needs to be introduced, use the following code:
### comm_type = [Communications[row]['Comm_Name'] for row in range(people)]
### the you'll need to update a few indicies as well...
recent_opens =  [rec_comms[row]['Email Opened'] for row in range(people)]
recent_nopens = [rec_comms[row]['Email Not Opened'] for row in range(people)]
# Go through all your raw data
#for entry in raw_data:
  # Get the fields you want
#  comm_type = entry["comm_type"]
#student = entry["student"]

# See if you've already started tracking this type of comm, if not, start tracking it.
#if comm_type not in each_comm_type:
#  each_comm_type[comm_type] = Counter()

# Record the data
#each_comm_type[comm_type][student] += 1

#return each_comm_type

###Coming up with a rudimentary scoring system...
###Since I cannot see if bulk is opened, well put low points into each sent bulk
###Find relative frequency of events to make points...
###Data seems odd, it must be that Incoming calls are calls from students, but Incoming mail is mail sent from Dubuque
###Dataframe of names with their comm statistics
Wrestlers = pd.DataFrame({'Name': names, 'Opened_Email': opens, 'Not_Opened_Email': nopens, 'SMS': smss, 'Outgoing_Calls': outcall, 'Incoming_Email': inmail, 'Outgoing_Email': outmail, 'Incoming_Call': incall, 'On-Campus Visit': visits, 'General': general, 'Voice Mail': voice, 'Off-Campus Contact' : offcon, 'Evaluation' : eval, 'Postal Mail' : post})
### Here there will need to be new comm names added if you want additional types of comms
#Student Comm Score for individual comms: +1 for email opened * 1/Time, +3 for email sent or text * 1/Time, +5 for call *1/Time, +10 for visit. -1 for email not opened
#Coach Comm Score for individual comms:  +1 for each email sent, +3 text * 1/Time, +5 for call * 1/Time
tot_cont = Wrestlers.sum(axis = 1, numeric_only = True)
tot_cont_root = tot_cont**.5
###Here are the linear points for each type of communication. Just change the values that are currently 0, 1, 3 or 5 currently
for item in range(data.shape[0]):###dont change this 0
  old = (data['Last_Comm'][item]) >= 30
  if data['Contact_Type'][item] == 'Email Opened': #+1 Student and Coach
    S_Comm[item] = (1  - old)
    C_Comm[item] = (1  - old)
  if data['Contact_Type'][item] == 'Email Not Opened': #-1 Student, +1 Coach
    S_Comm[item] = 0
    C_Comm[item] = (1 - old*0.5)
  #if data['Contact_Type'][item] == 'SMS': #Both 3
  #  S_Comm[item] = data['SMSTally'][item] * (3 - old)
  #  C_Comm[item] = data['SMSTally'][item] * (2 - old)
  if data['Contact_Type'][item] == 'General': #Both 1
    S_Comm[item] = ( 1 - old)
    C_Comm[item] = (1 - old)
  if data['Contact_Type'][item] == 'Incoming Call': #Student Only
    S_Comm[item] = (5 - old*2.5)###this 2.5 is a weight the penalizes older comms
    C_Comm[item] = (0)
  if data['Contact_Type'][item] == 'Incoming Email': #Student Only
    S_Comm[item] = (3 - old)
    C_Comm[item] = (0)
  if data['Contact_Type'][item] == 'Official On-Campus Visit': #Student Only
    S_Comm[item] = (10)
    C_Comm[item] = (0)
  if data['Contact_Type'][item] == 'Outgoing Call': #Coach Only
    S_Comm[item] = (0)
    C_Comm[item] = (5 - old*3)###this 3 is a weight the penalizes older comms
  if data['Contact_Type'][item] == 'Outgoing Email': #Coach Only
    S_Comm[item] = (0)
    C_Comm[item] = (3 - old)
  if data['Contact_Type'][item] == 'Voice Mail': #Coach Only
    S_Comm[item] = (0)
    C_Comm[item] = (5 - old*3)###this 3 is a weight the penalizes older comms
data['S_Comm'] = S_Comm
data['C_Comm'] = C_Comm

### ADD NEW ITEMS TO WRESTLERS AS NEEDED ###
###1:Any Communication
time_since = []
time_since_open = []
s_comm = []
c_comm = []
cumsum = list(np.cumsum(tot_cont))
cumsum.insert(0, 0)
RecentOpen = [round(a / (a+b + 0.0001), 2) for a,b in zip(recent_opens, recent_nopens)]
for person in range(people):
  Ssms_score = RecentOpen[person]*round(sum(data[data['Contact_Type'] == "SMS"].S_Comm[cumsum[person]:cumsum[person+1]], 2))
  Csms_score = (1-RecentOpen[person])*round(sum(data[data['Contact_Type'] == "SMS"].C_Comm[cumsum[person]:cumsum[person+1]], 2))
###Change the sSMS to something less black and white. Also normalize more for recent communications.
###Perhaps as Recent open approaches 100%, Sssm approaches 50%?
  mixi = min(data.Last_Comm[cumsum[person]:cumsum[person+1]])
  time_since.append(mixi)
  mixo = min(data.Last_Open[cumsum[person]:cumsum[person + 1]])
  time_since_open.append(mixo)
  c_dot = sum(data.C_Comm[cumsum[person]:cumsum[person+1]])
  s_dot = sum(data.S_Comm[cumsum[person]:cumsum[person+1]])
  s_comm.append(round(s_dot + Ssms_score, 2))
  c_comm.append(round(c_dot + Csms_score, 2))

###2:opened - 3*not opened
RAW1 = (Wrestlers.sum(axis = 1, numeric_only = True) * (Wrestlers.Opened_Email / (Wrestlers.Opened_Email + Wrestlers.Not_Opened_Email)) + Wrestlers.SMS + Wrestlers.Outgoing_Calls + 2 * Wrestlers.Incoming_Call + 2 * Wrestlers.Outgoing_Email + Wrestlers.General + 5 * Wrestlers['On-Campus Visit'])
###3:opened - 3*not opened + SMS + Calls
RAW2 = Wrestlers.Opened_Email - 3 * Wrestlers.Not_Opened_Email + Wrestlers.SMS + Wrestlers.Outgoing_Calls + 5 * Wrestlers.Incoming_Call + 5 * Wrestlers.Outgoing_Email

RAW3 = round((tot_cont_root * (Wrestlers.Opened_Email / (Wrestlers.Opened_Email + Wrestlers.Not_Opened_Email)) + Wrestlers.SMS + Wrestlers.Outgoing_Calls + 2 * Wrestlers.Incoming_Call + 2 * Wrestlers.Outgoing_Email + Wrestlers.General + 5 * Wrestlers['On-Campus Visit']), 2)
#RAW3DF = pd.DataFrame({'RAW3': RAW3, 'Name': names})
#RAW3DF.to_csv('Formula3CommScoresApr.csv')
OpenPer = round((Wrestlers.Opened_Email + 0.001) / (Wrestlers.Opened_Email + Wrestlers.Not_Opened_Email + 0.002), 2)
 #Mail open percent always slightly above zero to avoid error down the line

CommWeight = (0.5)*(OpenPer ** 0.5 + 1)#flattening function for communication weight for those who tend to communicate less via email
f1min = min(RAW1)
f1max = max(RAW1)
SCORE1 = round(10*RAW1/f1max,2)
Wrestlers['Total Contacts'] = tot_cont
Wrestlers['Mail Percent'] = OpenPer
Wrestlers['RAW1'] = round(RAW1,2)
Wrestlers['RAW2'] = round(RAW2, 2)
Wrestlers['RAW3'] = round(RAW3, 2)
Wrestlers['Time_since'] = time_since
Wrestlers['Time_since_Open'] = time_since_open
Wrestlers['SCORE1'] = SCORE1
Wrestlers['S_COMM'] = [round(a*b, 2) for a,b in zip(s_comm, CommWeight)]
#Wrestlers['S_COMM'] = s_comm
Wrestlers['C_COMM'] = c_comm
Wrestlers['Recent Open Percent'] = RecentOpen

Wrestlers.to_csv('Wrestling_Comm_Scores_6-01-20.csv')
#Below is just a vizualization
"""
N = 40
newdat = pd.read_csv('Student_Coach_Data.csv')
newdat = newdat[0:40]
x = newdat['NORMALIZED']
y = newdat['NORMALIZED.1']
area = ((x+y))**2
theta = np.arange(0, 2*np.pi, 0.01)
area1 = np.ma.masked_where( 12/(x+y) > 1, area)
plt.scatter(x,y, s=area1)
plt.plot(5*np.cos(theta)+12, 5*np.sin(theta)+12)
plt.plot(7*np.cos(theta)+12, 7*np.sin(theta)+12)
plt.axis('equal')
plt.xlim(5, 11)
plt.ylim(5, 11)
plt.savefig('plot2.png')
plt.clf()
"""