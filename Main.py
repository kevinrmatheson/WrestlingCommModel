import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
data = pd.read_csv("AllCommAugDec.csv") #Just make sure useing most recent file

data = data.iloc[:,0:5]
year = 2019
mo = 12
day = 31
today = (year - 1) * 365 + (mo-1) * 30 + (day)
columns = ['Wrestler', 'Grad_Year', 'Contact_Type', 'Contact_By', 'Date']
data.columns = columns
days = []
for row, comm in data.iterrows():
  text = comm.Date
  mo1 = re.findall(r'\d+', text)[0]
  day1 = re.findall(r'\d+', text)[1]
  year1 = re.findall(r'\d+', text)[2]
  days.append((int(year1) - 1) * 365 + (int(mo1)-1) * 30 + (int(day1)))


data['Last_Comm'] =  [today - day for day in days]#approx last communcation
Avg_Last_Comm = np.mean(data.Last_Comm)

Comms = Counter(data.loc[:,'Contact_Type']).keys()
Comm_Nums = Counter(data.loc[:,'Contact_Type']).values()

names = data.Wrestler.unique()
Communications = []#dictionary of different communication types and amounts
Comm_Amount = []

for name in names:
  Communications.append(Counter(data[data.Wrestler == name].loc[:,'Contact_Type']))
  Comm_Amount.append(sum(list(Counter(data[data.Wrestler == name].loc[:,'Contact_Type']).values())))

people = len(names)

opens = [Communications[row]['Email Opened'] for row in range(people)]
nopens = [Communications[row]['Email Not Opened'] for row in range(people)]
smss = [Communications[row]['SMS'] for row in range(people)]
outcall = [Communications[row]['Outgoing Call'] for row in range(people)]
inmail = [Communications[row]['Incoming Email'] for row in range(people)]
outmail = [Communications[row]['Outgoing Email'] for row in range(people)]
incall = [Communications[row]['Incoming Call'] for row in range(people)]
visits = [Communications[row]['Official On-Campus Visit'] for row in range(people)]
general = [Communications[row]['General'] for row in range(people)]

Open_Mail = list(Comm_Nums)[0]#76 personal mail opened
N_Open_Mail = list(Comm_Nums)[5]#24 personal mail not opened
Inc_Mail = list(Comm_Nums)[1]#47
Out_Mail = list(Comm_Nums)[4]#2
Out_Call = list(Comm_Nums)[2]#32
SMS = list(Comm_Nums)[3]#52
Inc_Call = list(Comm_Nums)[6]#2

E_Open_P = round(Open_Mail / (Open_Mail + N_Open_Mail), 2) #Percent of Personalized mail opened
Avg_Comm = round(sum(list(Comm_Nums))/len(names),2)#Average number of communications to any student
Avg_P_Comm = round((Open_Mail + N_Open_Mail)/len(names),2)#Average number of personalized comms to any student
###Coming up with a rudimentary scoring system...
###Since I cannot see if bulk is opened, well put low points into each sent bulk
###Find relative frequency of events to make points...
###Data seems odd, it must be that Incoming calls are calls from students, but Incoming mail is mail sent from Dubuque
###Dataframe of names with their comm statistics
Wrestlers = pd.DataFrame({'Name': names, 'Opened_Email': opens, 'Not_Opened_Email': nopens, 'SMS': smss, 'Outgoing_Calls': outcall, 'Incoming_Email': inmail, 'Outgoing_Email': outmail, 'Incoming_Call': incall, 'On-Campus Visit': visits, 'General': general})


###1:Any Communication
tot_cont = Wrestlers.sum(axis = 1, numeric_only = True)
###2:opened - 3*not opened
form1 = (Wrestlers.sum(axis = 1, numeric_only = True) * (Wrestlers.Opened_Email / (Wrestlers.Opened_Email + Wrestlers.Not_Opened_Email)) + Wrestlers.SMS + Wrestlers.Outgoing_Calls + 2 * Wrestlers.Incoming_Call + 2 * Wrestlers.Outgoing_Email + Wrestlers.General + 5 * Wrestlers['On-Campus Visit'])**0.5
###3:opened - 3*not opened + SMS + Calls
form2 = Wrestlers.Opened_Email - 3 * Wrestlers.Not_Opened_Email + Wrestlers.SMS + Wrestlers.Outgoing_Calls + 5 * Wrestlers.Incoming_Call + 5 * Wrestlers.Outgoing_Email

time_since = []
cumsum = list(np.cumsum(tot_cont))
cumsum.insert(0, 0)
for person in range(people):
  mixi = min(data.Last_Comm[cumsum[person]:cumsum[person+1]])
  time_since.append(mixi)
f1min = min(form1)
f1max = max(form1)
f1norm = round(10*form1/f1max,2)

Wrestlers['Total Contacts'] = tot_cont
Wrestlers['Mail Percent'] = round(Wrestlers.Opened_Email / (Wrestlers.Opened_Email + Wrestlers.Not_Opened_Email), 2)
Wrestlers['Formula1'] = round(form1,2)
Wrestlers['Formula2'] = form2
Wrestlers['Time_since'] = time_since
Wrestlers['F1Norm'] = f1norm
Wrestlers.to_csv('Wrestling_Comm_Scores_Dec.csv')
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