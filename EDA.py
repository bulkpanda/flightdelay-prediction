# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:12:28 2021

@author: Kunal Patel
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

dataset=pd.read_csv('FlightDelays.csv')
sns.countplot(x='Flight Status', data = dataset)
#============================================================================
dataset['flight_status'] = pd.get_dummies(dataset['Flight Status'])['ontime']

#====================Analysis on Carrier=====================================
carrier = dataset[['CARRIER','flight_status']]
carrier_encode = pd.get_dummies(carrier['CARRIER'])
headers = carrier_encode.columns
carrier_per= np.zeros(len(headers)) 

for i in range (len(carrier_per)):
    i_carrier_data = carrier.loc[carrier['CARRIER'] == headers[i]]
    TotalFlights=len(i_carrier_data)
    carrier_per[i] = i_carrier_data['flight_status'].sum() #Total ontime flights
    carrier_per[i] = carrier_per[i] *100/TotalFlights #% of ontime flights
    carrier_per[i] = 100-carrier_per[i] #% of delayed flights
    


carriers = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(carriers,carrier_per)
plt.xlabel('Carriers')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By Carriers')
plt.savefig('EDACarriers')
plt.show()
print(carrier_per)
#==============================================================================

#====================Analysis on Dept_time=====================================
dept_time = dataset[['DEP_TIME','flight_status']]
dept_time_encode = pd.get_dummies(dept_time['DEP_TIME'])
headers = dept_time_encode.columns
dept_time_per= np.zeros(len(headers)) 
for i in range (len(dept_time_per)):
    i_dept_time_data = dept_time.loc[dept_time['DEP_TIME'] == headers[i]]
    TotalFlights=len(i_dept_time_data)
    dept_time_per[i] = i_dept_time_data['flight_status'].sum() #Total ontime flights
    dept_time_per[i] = dept_time_per[i] *100/TotalFlights #% of ontime flights
    dept_time_per[i] = 100-dept_time_per[i] #% of delayed flights
    
#print(dept_time_per)

dept_time = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(dept_time,dept_time_per)
plt.xlabel('Carriers')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By Departure Time')
plt.savefig('EDADeptTime')
plt.show()
#==============================================================================

#====================Analysis on Dept_time=====================================
# dept_time = dataset[['CRS_DEP_TIME','flight_status']]
# dept_time_encode = pd.get_dummies(dept_time['CRS_DEP_TIME'])
# headers = dept_time_encode.columns
# dept_time_per= np.zeros(len(headers)) 

# for i in range (len(dept_time_per)):
#     i_dept_time_data = dept_time.loc[dept_time['CRS_DEP_TIME'] == headers[i]]
#     TotalFlights=len(i_dept_time_data)
#     dept_time_per[i] = i_dept_time_data['flight_status'].sum() #Total ontime flights
#     dept_time_per[i] = dept_time_per[i] *100/TotalFlights #% of ontime flights
#     dept_time_per[i] = 100-dept_time_per[i] #% of delayed flights
    
# #print(dept_time_per)

# dept_time = [i for i in headers]
# #fig1 = plt.figure()
# #ax = fig1.add_axes([0,0,1,1])
# plt.bar(dept_time,dept_time_per)
# plt.xlabel('Carriers')
# plt.ylabel('(%) of Flights Delayed')
# plt.title('Flight Delays By Departure Time')
# plt.savefig('EDADeptTime')
# plt.show()
# print(dept_time_per)
#==============================================================================

#====================Analysis on Destination===================================
dest = dataset[['DEST','flight_status']]
dest_encode = pd.get_dummies(dest['DEST'])
headers = dest_encode.columns
dest_per= np.zeros(len(headers)) 

for i in range (len(dest_per)):
    i_dest_data = dest.loc[dest['DEST'] == headers[i]]
    TotalFlights=len(i_dest_data)
    dest_per[i] = i_dest_data['flight_status'].sum() #Total ontime flights
    dest_per[i] = dest_per[i] *100/TotalFlights #% of ontime flights
    dest_per[i] = 100-dest_per[i] #% of delayed flights
    


destinations = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(destinations,dest_per)
plt.xlabel('Destination')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By Destination')
plt.savefig('EDADestination')
plt.show()
print(dest_per)
#==============================================================================

#====================Analysis on Distance=====================================

distance = dataset[['DISTANCE','flight_status']]
distance_encode = pd.get_dummies(distance['DISTANCE'])
headers = distance_encode.columns
distance_per= np.zeros(len(headers)) 

for i in range (len(distance_per)):
    i_distance_data = distance.loc[distance['DISTANCE'] == headers[i]]
    TotalFlights=len(i_distance_data)
    distance_per[i] = i_distance_data['flight_status'].sum() #Total ontime flights
    distance_per[i] = distance_per[i] *100/TotalFlights #% of ontime flights
    distance_per[i] = 100-distance_per[i] #% of delayed flights
    


distances = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(distances,distance_per)
plt.xlabel('distance')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By distance')
plt.savefig('EDAdistance')
plt.show()
print(distance_per)
#==============================================================================

#====================Analysis on Origin=====================================

origin = dataset[['ORIGIN','flight_status']]
origin_encode = pd.get_dummies(origin['ORIGIN'])
headers = origin_encode.columns
origin_per= np.zeros(len(headers)) 

for i in range (len(origin_per)):
    i_origin_data = origin.loc[origin['ORIGIN'] == headers[i]]
    TotalFlights=len(i_origin_data)
    origin_per[i] = i_origin_data['flight_status'].sum() #Total ontime flights
    origin_per[i] = origin_per[i] *100/TotalFlights #% of ontime flights
    origin_per[i] = 100-origin_per[i] #% of delayed flights
    


origins = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(origins,origin_per)
plt.xlabel('origin')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By origin')
plt.savefig('EDAorigin')
plt.show()
print(origin_per)
#==============================================================================

#====================Analysis on Weather=====================================

weather = dataset[['Weather','flight_status']]
weather_encode = pd.get_dummies(weather['Weather'])
headers = weather_encode.columns
weather_per= np.zeros(len(headers)) 

for i in range (len(weather_per)):
    i_weather_data = weather.loc[weather['Weather'] == headers[i]]
    TotalFlights=len(i_weather_data)
    weather_per[i] = i_weather_data['flight_status'].sum() #Total ontime flights
    weather_per[i] = weather_per[i] *100/TotalFlights #% of ontime flights
    weather_per[i] = 100-weather_per[i] #% of delayed flights
    


weathers = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(weathers,weather_per)
plt.xlabel('weather')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By weather')
plt.savefig('EDAweather')
plt.show()
print(weather_per)
#==============================================================================

#====================Analysis on week days=====================================
Weekdays = dataset[['DAY_WEEK','flight_status']]
Weekdaysper= np.zeros(7) 

for i in range (0,7):
    i_week_data = Weekdays.loc[Weekdays['DAY_WEEK'] == i+1]
    TotalFlights=len(i_week_data)
    Weekdaysper[i] = i_week_data['flight_status'].sum() #Total ontime flights
    Weekdaysper[i] = Weekdaysper[i] *100/TotalFlights #% of ontime flights
    Weekdaysper[i] = 100-Weekdaysper[i] #% of delayed flights
    


days = ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun')
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(days,Weekdaysper)
plt.xlabel('Days of the week')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By Weekdays')
plt.savefig('EDAWeekdays')
plt.show()
print(Weekdaysper)
#==============================================================================

#====================Analysis on Day of Month==================================
DOM = dataset[['DAY_OF_MONTH','flight_status']]
DOM_encode = pd.get_dummies(DOM['DAY_OF_MONTH'])
headers = DOM_encode.columns
DOM_per= np.zeros(len(headers)) 

for i in range (len(DOM_per)):
    i_DOM_data = DOM.loc[DOM['DAY_OF_MONTH'] == headers[i]]
    TotalFlights=len(i_DOM_data)
    DOM_per[i] = i_DOM_data['flight_status'].sum() #Total ontime flights
    DOM_per[i] = DOM_per[i] *100/TotalFlights #% of ontime flights
    DOM_per[i] = 100-DOM_per[i] #% of delayed flights
    


DOMs = [i for i in headers]
#fig1 = plt.figure()
#ax = fig1.add_axes([0,0,1,1])
plt.bar(DOMs,DOM_per)
plt.xlabel('DOMs')
plt.ylabel('(%) of Flights Delayed')
plt.title('Flight Delays By DOM')
plt.savefig('EDADOM')
plt.show()
print(DOM_per)
#==============================================================================
