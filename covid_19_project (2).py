# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import folium
import io
from operator import itemgetter, attrgetter
import json
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from google.colab import drive ,files
import sys
from difflib import SequenceMatcher
from fbprophet import Prophet

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

drive.mount('/content/gdrive', force_remount=True)
sys.path.append('/content/gdrive/My Drive')

os.chdir("/content/gdrive/My Drive")
data = pd.read_csv('WHO-COVID-19-global-data.csv')
data=pd.DataFrame(data)
population =  pd.read_excel('WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx', index_col=None, header=None)
population = pd.DataFrame(population)
population = np.array(population)
more_data = pd.read_csv('owid-covid-data.csv')
more_data = pd.DataFrame(more_data)
sys.path.append('/content/gdrive/My Drive')

#print((total_data['iso_code']))
new_tests=more_data['new_tests']
total_tests=more_data['total_tests']
hosp_patients=more_data['hosp_patients']
icu_patients=more_data['icu_patients']
all_countries=more_data['location'].unique()
#country_codes=total_data[' Country_code'].unique()
Dates=more_data['date'].unique()
Dates=sorted(list (Dates))
print(len(Dates))
print(len(all_countries))

icu_patients=[]
new_tests=[]
cumulative_tests=[]
hosp_patients=[]
for i in all_countries:
  icu_patients.append(list(more_data['icu_patients'].where(more_data['location']==i).dropna()))
  hosp_patients.append(list(more_data['hosp_patients'].where(more_data['location']==i).dropna()))
  cumulative_tests.append(list(more_data['total_tests'].where(more_data['location']==i).dropna()))
  new_tests.append(list(more_data['new_tests'].where(more_data['location']==i).dropna()))

countries=data[' Country'].unique()
country_codes=data[' Country_code'].unique()
who_region=data[' WHO_region'].unique()
Dates_reported=data['Date_reported'].unique()
Dates_reported =sorted(list (Dates_reported))
print(len(Dates_reported))

New_cases=[]
New_deaths=[]
Cumulative_cases=[]
Cumulative_deaths=[]
for c in countries:
    New_cases.append(list(data[' New_cases'].where(data[' Country']==c).dropna()))
    New_deaths.append(list(data[' New_deaths'].where(data[' Country']==c).dropna()))
    Cumulative_cases.append(list(data[' Cumulative_cases'].where(data[' Country']==c).dropna()))
    Cumulative_deaths.append(list(data[' Cumulative_deaths'].where(data[' Country']==c).dropna()))

print(len(New_cases[0]))

def World_Population(pop):
  tlist = []
  for i in range(43,len(pop)):
    for j in range(0,len(countries)):
      if (pop[i,2]==countries[j]):
        mlk =[pop[i,2],pop[i,-1]*1000]
        tlist.append(mlk)
  return (tlist)

Population = World_Population(population)
print(Population)


def Make_Mrcountries(k):
  MRlist = []
  for i in range(len(Cumulative_cases[k])):
    if (Cumulative_cases[k][i]!=0):
      MRlist.append((100*Cumulative_deaths[k][i])/Cumulative_cases[k][i])
    else:
      MRlist.append(0)
  return (MRlist)

def Make_Overall(alist):
  countries[135] = "Micronesia"
  for i in range(len(countries)):
      templist = []
      My_array = ["empty",0,0,0,0,0,[],[],[],[],[],0]
      My_array[0] = countries[i]
      My_array[1] = Cumulative_cases[i][-1]
      My_array[2] = Cumulative_deaths[i][-1]
      if (Cumulative_cases[i][-1]!=0):
        My_array[3] = (Cumulative_deaths[i][-1]/Cumulative_cases[i][-1])*100
      else:
        My_array[3] = 0.0
      for j in Population:
        if (j[0]==countries[i]):
          My_array[4] = ((Cumulative_cases[i][-1])/j[1])*1000000
          My_array[5] = ((Cumulative_deaths[i][-1])/j[1])*1000000
          break
        else:
          My_array[4] = 0.0
          My_array[5] = 0.0
      My_array[6] = Make_Mrcountries(i)
      for j in range(len(all_countries)):
        x = similar(all_countries[j],countries[i])
        templist.append(x)
      if (countries[i]=="Czechia"):
        j=49
      elif (countries[i]=="Russian Federation"):
        j=163
      else:
        j = templist.index(max(templist))
      My_array[7] = hosp_patients[j]
      My_array[8] = icu_patients[j]
      My_array[9] = new_tests[j]
      My_array[10] = cumulative_tests[j]
      if (My_array[10]!=[]):
        My_array[11] = My_array[10][-1]
      else:
        My_array[11] = 0
      alist.append(My_array)
  return (alist)

Overall_per_country = []
Overll_per_country = Make_Overall(Overall_per_country)

for i in range(len(countries)):
  templist=[]
  for j in range(len(all_countries)):
    x = similar(all_countries[j],countries[i])
    templist.append(x)
  j = templist.index(max(templist))
  print(countries[i], "is similar to", all_countries[j])

for i in range(len(all_countries)):
  print(all_countries[i],i)
  if (cumulative_tests[i]!=[]):
    print((hosp_patients[i]))
    print((icu_patients[i]))
    print("")

L=len(Dates_reported)
def padding_function(cases,deaths,cum_cases,cum_deaths,newtests,cumtests,hospat,icupat,L):
  for i,c in enumerate(countries):
    P=L-len(cases[i])
    for k in range(P):
      cases[i].insert(0,0)
      deaths[i].insert(0,0)
      cum_cases[i].insert(0,0)
      cum_deaths[i].insert(0,0)
  for i,c in enumerate(all_countries):
    P1=L-len(newtests[i])
    P2=L-len(cumtests[i])
    P3=L-len(hosp_patients[i])
    for k in range(P1):
      newtests[i].insert(0,0)
    for k in range(P2):
      cumtests[i].insert(0,0)
    for k in range(P3):
      hospat[i].insert(0,0)
      icupat[i].insert(0,0)
  return cases,deaths,cum_cases,cum_deaths,newtests,cumtests,hospat,icupat

New_cases,New_deaths,Cumulative_cases,Cumulative_deaths,new_tests,cumulative_tests,hosp_patients,icu_patients = padding_function(New_cases,New_deaths,Cumulative_cases,Cumulative_deaths,new_tests,cumulative_tests,hosp_patients,icu_patients,L) 
print(len(icu_patients[97]))

def moving_av(series,w):
    step=w//2
    s=[]
    for i in range(step):
        s.append(0)
    for c in range(step,len(series)-step):
        sum=0
        for i in range(c-step,c+step+1):
            sum=sum+series[i]
        s.append(sum/w)
    for i in range(step):
        s.append(0)
    return s

def World_data():
    SumCases = 0
    SumDeaths = 0
    SumPop = 0
    for i in Population:
      SumPop = SumPop + i[1]
    for i in Overall_per_country:
      SumCases = SumCases + i[1]
      SumDeaths = SumDeaths + i[2]
    W_c_p_m = int ((SumCases/SumPop)*1000000)
    W_d_p_m = int ((SumDeaths/SumPop)*1000000)
    day_list=[]
    time_list=[]
    clist_total = [sum(x) for x in zip(*Cumulative_cases)]
    dlist_total = [sum(x) for x in zip(*Cumulative_deaths)]
    clist_daily = [sum(x) for x in zip(*New_cases)]
    dlist_daily = [sum(x) for x in zip(*New_deaths)]
    for i,j in zip(clist_daily,dlist_daily):
      if (i!=0): day_list.append(j/i)
      else: day_list.append(0)
    
    for i,j in zip(clist_total,dlist_total):
      if (i!=0): time_list.append(j/i)
      else: time_list.append(0)

    return (clist_total,dlist_total,clist_daily,dlist_daily,day_list,time_list,SumCases,SumDeaths,W_c_p_m,W_d_p_m)

World_cases ,World_deaths, Daily_World_cases, Daily_World_deaths,M_R_global_per_day,M_R_global_over_time,Total_cases,Total_deaths,Wcases_per_million,Wdeaths_per_million = World_data()
World_cases = [round(x) for x in World_cases]
World_deaths = [round(x) for x in World_deaths]

def plot_top10(tlist,index,string1,string2,f):
  sorted_list = sorted(tlist, key=itemgetter(index),reverse=True)
  fig = plt.figure(figsize=(16,6))
  ax = fig.add_axes([0,0,1,1])
  plt.title("Top ten countries " +  string1,fontsize=20, color='C0')
  plt.xlabel('Country', fontsize=20)
  plt.ylabel(string2,fontsize=20)
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'yellow', 'magenta', 'green', 'orange', 'pink')
  for i in range(0,10):
    ax.bar(sorted_list[i][0],sorted_list[i][index],color=colors[i], width=0.9, label=sorted_list[i][0])
  plt.legend(fontsize=f)
  plt.show()

def plot_WorldMR(tlist,string):
  plt.figure(figsize=(50,20))
  plt.title("Worldwide {} Mortality rate of Covid-19".format(string),fontsize=50)
  plt.xlabel('Days passed since the begining', fontsize=40)
  plt.ylabel('{} mortality rate'.format(string),fontsize=30)
  plt.plot(Dates_reported, tlist, 'r', marker='o', linewidth=1.9, label='Mortaity Rate')
  plt.xticks(rotation=90)
  plt.yticks(size=20)
  plt.legend(loc='best',fontsize=40)
  plt.grid(True, linewidth=2.0)
  plt.show()

def plot_daily(cases,deaths,weekly_cases,weekly_deaths,country):
    fig = plt.figure(figsize=(50,20))
    ax = fig.add_axes([0,0,1,1])
    plt.title("Daily cases and deaths {}".format(country),fontsize=50)
    plt.xlabel('Day', fontsize=40)
    plt.ylabel('New Cases/New Deaths',fontsize=40)
    ax.bar(Dates_reported,cases,color='r',width=0.9, label='New Cases')
    ax.bar(Dates_reported,deaths, color= 'm',width=0.9, label='New Deaths')
    plt.plot(Dates_reported,weekly_cases,color='b',linewidth=6,label='weekly average cases')
    plt.plot(Dates_reported,weekly_deaths,color='g',linewidth=6,label='weekly average deaths')
    plt.yticks(size=20)
    plt.xticks(rotation=90)
    plt.legend(fontsize=40)
    plt.grid(True)
    plt.show()

def plot_cumulative(cases,deaths,country,scale,number):
    plt.figure(figsize=(50,20))
    plt.title("Cumulative cases and deaths {} ({} scale)".format(country,scale),fontsize=50)
    plt.xlabel('Days passed since the begining', fontsize=40)
    plt.ylabel('Cumulative Cases/Deaths ('+ scale + ' scale)',fontsize=30)
    plt.plot(Dates_reported, cases, 'r', marker='o', linewidth=1.9, label='Cases')
    plt.plot(Dates_reported, deaths, 'm', marker='*', linewidth=1.9, label='Deaths')
    plt.yscale(scale)
    plt.xticks(rotation=90)
    if ((scale=="linear") and (number!=0)):
      d = (cases[-1]-cases[0])/10
      plt.yticks(np.arange(cases[0], cases[-1], d))
    plt.legend(loc='best',fontsize=40)
    plt.grid(True, linewidth=2.0)
    plt.show()

def plot_daily_tests(tests,weekly_tests,country):
    fig = plt.figure(figsize=(50,20))
    ax = fig.add_axes([0,0,1,1])
    plt.title("Daily tests {}".format(country),fontsize=50,color='C0')
    plt.xlabel('Day', fontsize=40)
    plt.ylabel('New Tests',fontsize=40)
    ax.bar(Dates_reported,tests,color='g',width=0.9, label='New Tests')
    plt.plot(Dates_reported,weekly_tests,color='b',linewidth=6,label='weekly average tests')
    plt.yticks(size=20)
    plt.xticks(rotation=90)
    plt.legend(fontsize=40)
    plt.grid(True)
    plt.show()

def plot_cum_tests(tests,country,scale,number):
    plt.figure(figsize=(50,20))
    plt.title("Cumulative tests {} ({} scale)".format(country,scale),fontsize=50,color='C0')
    plt.xlabel('Days passed since the begining', fontsize=40)
    plt.ylabel('Cumulative Tests ('+ scale + ' scale)',fontsize=30)
    plt.plot(Dates_reported, tests, 'g', marker='*', linewidth=1.9, label='Tests')
    plt.yscale(scale)
    plt.xticks(rotation=90)
    if ((scale=="linear") and (number!=0)):
      d = (tests[-1]-tests[0])/10
      plt.yticks(np.arange(tests[0], tests[-1], d),size=10)
    plt.legend(loc='best',fontsize=40)
    plt.grid(True, linewidth=2.0)
    plt.show()

def plot_patients(hosp_p,icu_p,country,scale,number):
    plt.figure(figsize=(50,20))
    plt.title("Hospitilized patients and ICU patients {} ({} scale)".format(country,scale),fontsize=50,color='C0')
    plt.xlabel('Days passed since the begining', fontsize=40)
    plt.ylabel('Number of patients ('+ scale + ' scale)',fontsize=30)
    plt.plot(Dates_reported, hosp_p, 'c', marker='o', linewidth=1.9, label='Hospitilized Patients')
    plt.plot(Dates_reported, icu_p, 'r', marker='*', linewidth=1.9, label='ICU Patients')
    plt.yscale(scale)
    plt.xticks(rotation=90)
    if ((scale=="linear") and (number!=0)):
      d = (hosp_p[-1]-hosp_p[0])/10
      plt.yticks(np.arange(hosp_p[0],hosp_p[-1], d),size=10)
    plt.legend(loc='best',fontsize=40)
    plt.grid(True, linewidth=2.0)
    plt.show()

weekly_tests=moving_av(new_tests[199],7)
plot_daily_tests(new_tests[199],weekly_tests,"in "+all_countries[199])
plot_cum_tests(cumulative_tests[199],"in "+all_countries[199],"linear",cumulative_tests[14][-1])

plot_patients(hosp_patients[97],icu_patients[97],"in "+all_countries[97],"linear",hosp_patients[97][-1])

def plot_pie_charts(tlist, index, title):
      S = 0
      xlist = []
      ylist = []
      sorted_list = sorted(tlist, key=itemgetter(index),reverse=True)
      for i in range(0,11):
        if (i!=10):
          xlist.append(sorted_list[i][index])
          ylist.append(sorted_list[i][0])
        else:
          for j in range(i,len(countries)):
            S = S + sorted_list[j][index]
          xlist.append(S)
          ylist.append('Others')

      c = ['lightcoral', 'rosybrown', 'sandybrown', 'navajowhite', 'gold', 'khaki', 'lightskyblue', 'turquoise', 'lightslategrey', 'thistle', 'pink']
      plt.figure(figsize=(16,14))
      plt.title(title, size=16)
      plt.pie(xlist, colors=c,shadow=True, labels=xlist)
      plt.legend(ylist, loc='upper left', fontsize=11)
      plt.show()

def plot_top5(tlist1,tlist2,index1,index2,string):
    sorted_list = sorted(tlist1,key=itemgetter(index1),reverse=True)
    name_list = sorted(tlist2,key=itemgetter(index2),reverse=True)
    plt.figure(figsize=(50,20))
    plt.title("Cumulative {} in the top 5 countries".format(string),fontsize=45)
    plt.xlabel('Days passed since the begining', fontsize=40)
    plt.ylabel("Cumulative {}".format(string),fontsize=30)
    for i in range(0,5):
      plt.plot(Dates_reported,sorted_list[i], linewidth=3.9, label=name_list[i][0])
    plt.xticks(rotation=90)
    plt.yticks(size=20)
    plt.legend(loc='best',fontsize=30)
    plt.grid(True, linewidth=2.0)
    plt.show()

print("The total cases of COVID-19 are:",Total_cases)
print("The total deaths of COVID-19 are:",Total_deaths)
print("The World's Mortality Rate for COVID-19 is: {}%".format((Total_deaths/Total_cases)*100))
print("There are {} cases per 1 million and {} deaths per 1 million people all around the world".format(Wcases_per_million,Wdeaths_per_million))

plot_WorldMR(M_R_global_over_time,"total")
plot_WorldMR(M_R_global_per_day,"daily")

plot_top10(Overall_per_country,1,'with most recorded cases','Total Cases',15)
plot_top10(Overall_per_country,2,'with most recorded deaths','Total Deaths',15)
plot_top10(Overall_per_country,3,'by mortality rate','Mortality Rate',15)
plot_top10(Overall_per_country,4,'by cases/million of population','Cases per million',13)
plot_top10(Overall_per_country,5,'by deaths/million of population','Deaths per million',13)
plot_top10(Overall_per_country,11,'by most conducting test','Total Number of Tests',13)

#for i in range(len(countries)):
#  weekly_deaths=moving_av(New_deaths[i],7)
#  weekly_cases=moving_av(New_cases[i],7)
#  plot_daily(New_cases[i],New_deaths[i],weekly_cases,weekly_deaths,"in "+countries[i])

weekly_deaths=moving_av(Daily_World_deaths,7)
weekly_cases=moving_av(Daily_World_cases,7)
plot_daily(Daily_World_cases,Daily_World_deaths,weekly_cases,weekly_deaths,'Worldwide')

weekly_deaths=moving_av(Daily_World_deaths,7)
weekly_cases=moving_av(Daily_World_cases,7)
plot_daily(Daily_World_cases,Daily_World_deaths,weekly_cases,weekly_deaths,'Worldwide')

#for i in range(len(countries)):
#  plot_cumulative(Cumulative_cases[i],Cumulative_deaths[i],"in "+countries[i],"linear",Cumulative_cases[i][-1])
#  plot_cumulative(Cumulative_cases[i],Cumulative_deaths[i],"in "+countries[i],"log",Cumulative_cases[i][-1])

plot_cumulative(World_cases,World_deaths,"Worldwide","linear",1)
plot_cumulative(World_cases,World_deaths,"Worldwide","log",1)

plot_pie_charts(Overall_per_country, 1, 'Covid-19 Confirmed Cases per Country(Pie Chart)')
plot_pie_charts(Overall_per_country, 2, 'Covid-19 Confirmed Deaths per Country(Pie Chart)')

temp_list = sorted(Overall_per_country,key=itemgetter(1),reverse=True)
country_df = pd.DataFrame({'Country Name':[c[0] for c in temp_list], 'Number of Confirmed Cases': [int (c[1]) for c in temp_list],
                          'Number of Deaths': [int (c[2]) for c in temp_list],
                          'Mortality Rate': [c[3] for c in temp_list],
                           'Cases per million': [c[4] for c in temp_list],
                           'Deaths per million': [c[5] for c in temp_list]})

country_df.style.background_gradient(cmap='Oranges')

plot_top5(Cumulative_cases,Overall_per_country,-1,1,"Cases")
plot_top5(Cumulative_deaths,Overall_per_country,-1,2,"Deaths")

plt.figure(figsize=(64,64))
plt.rc('ytick',labelsize=16)
plt.title("Numbrer of confirmed cases per country around the globe", size=50)
plt.xlabel("Number of confirmed cases",size=40)
step = (1/2)*10**6
plt.xticks(np.arange(0, 10**7, step), size=32)
plt.grid(True)
for i in range(len(Cumulative_cases)):
  plt.barh(countries[i],Cumulative_cases[i][-1])
plt.show()

"""###Prediction for every country"""

for j in range(len(New_cases)):
  Sumlast10 = 0
  for i in range(len(New_cases[j]),len(New_cases[j])-8,-1):
    if ((len(New_cases[j])-i)<2):
      Sumlast10 = Sumlast10 + 2.5*New_cases[j][i-1]
    elif ((len(New_cases[j])-i)<4):
      Sumlast10 = Sumlast10 + 2*New_cases[j][i-1]
    elif ((len(New_cases[j])-i)<6):
      Sumlast10 = Sumlast10 + 0.5*New_cases[j][i-1]
    else:
      Sumlast10 = Sumlast10 + 0.4*New_cases[j][i-1]
  NextDayCases = Sumlast10/8
  print(countries[j], "will have around", round(NextDayCases), "cases in the 3th of November")

Passed_days_list = []
for i in range(len(Dates_reported)):
  Passed_days_list.append(i)
days_in_future = 10
future_forcast = np.array([i for i in range(len(Passed_days_list)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

start = '2020-01-03'
start_date = datetime.datetime.strptime(start, '%Y-%m-%d')
future_dates_reported = []
for i in range(len(future_forcast)):
    future_dates_reported.append((start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

#On X_train, X_test we have days and on y_train, y_test the total cases 
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(Passed_days_list, World_cases, test_size=0.15, shuffle=False) 
World_cases = np.array(World_cases).reshape(-1, 1)
Passed_days_list = np.array(Passed_days_list).reshape(-1, 1)
X_train_confirmed = np.array(X_train_confirmed).reshape(-1, 1)
X_test_confirmed = np.array(X_test_confirmed).reshape(-1, 1)
y_train_confirmed = np.array(y_train_confirmed).reshape(-1, 1)
y_test_confirmed = np.array(y_test_confirmed).reshape(-1, 1)

def plot_predictions(Real_cases,Predict_cases,Present_days,Future_days,color,string):
    plt.figure(figsize=(16,10))
    plt.plot(Present_days,Real_cases,color='blue')
    plt.plot(Future_days, Predict_cases, linestyle='dashed',color=color)
    plt.title("Numbrer of confirmed cases VS {}".format(string),size=25)
    plt.ylabel("Number of confirmed cases",size=15)
    plt.xlabel("Days passed since begging",size=15)
    plt.legend(['Confirmed Cases', string], fontsize=15)
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.grid(True)
    plt.show()

def test_predictor(predictor1,predictor2,string,flag):
  if (flag):
    predictor1.fit(X_train_confirmed,y_train_confirmed)
    test_pred = predictor1.predict(X_test_confirmed)
  else:
    X_new_test = predictor1.fit_transform(X_test_confirmed)
    test_pred = predictor2.predict(X_new_test)
  plt.plot(y_test_confirmed)
  plt.plot(test_pred)
  plt.grid(True)
  plt.legend(['Real Cases',string])

def Show_prediction_by_day(string,mypred):
  print(string,"for the next {} days (Total cases):".format(days_in_future))
  my_set = set (zip( future_dates_reported[-10:], np.round(mypred[-10:])))
  my_set = sorted(my_set)
  for x in my_set: 
    print(x[0]+":",x[1])
  print("For the next {} days the New cases will be according to {}:".format(days_in_future,string))
  for i in range(10,0,-1):
    cases_today = mypred[-i] - mypred[-(i+1)]
    print(future_dates_reported[-i] +":",  np.round(cases_today))

"""###SVM Prediction"""

SVΜ_cases = SVR(shrinking=True, kernel="poly", gamma=0.2, epsilon=0.5, degree=3, C=0.03, coef0=1.5)
World_cases.shape = (len(World_cases),)
SVΜ_cases.fit(Passed_days_list,World_cases)
predict_SVΜ = SVΜ_cases.predict(future_forcast) 
test_predictor(SVΜ_cases,None,'SVM Predictions', True)

plot_predictions(World_cases, predict_SVΜ, adjusted_dates, future_forcast, 'red','SVM Predictions')

Show_prediction_by_day('SVM predictions',predict_SVΜ)

"""###Linear Regression Prediction"""

linearR_cases = LinearRegression(fit_intercept=True, normalize=True)
linearR_cases.fit(Passed_days_list,World_cases)
predict_linearR = linearR_cases.predict(future_forcast) 
test_predictor(linearR_cases,None,'Linear Regression Predictions',True)

plot_predictions(World_cases, predict_linearR, adjusted_dates, future_forcast, 'green','Linear Regression Predictions')

Show_prediction_by_day('Linear Regression predictions',predict_linearR)

"""###Polynomial Regression Interpolation"""

poly_reg = PolynomialFeatures(degree=4)
X_poly_present = poly_reg.fit_transform(Passed_days_list)
X_poly_future = poly_reg.fit_transform(future_forcast)
lin_reg = LinearRegression()
lin_reg.fit(X_poly_present, World_cases)
predict_polynomialR = lin_reg.predict(X_poly_future)
test_predictor(poly_reg,lin_reg,'Polynomial Regression Predictions',False)

plot_predictions(World_cases, predict_polynomialR, adjusted_dates, future_forcast, 'orange','Polynomial Regression Predictions')

Show_prediction_by_day('Polynomial Regression predictions',predict_polynomialR)

"""###Bayesian Ridge Interpolation"""

bayesian_poly = PolynomialFeatures(degree=4)
X_new_present = bayesian_poly.fit_transform(Passed_days_list)
X_new_future = bayesian_poly.fit_transform(future_forcast)
bayesian = BayesianRidge(n_iter=600,compute_score=True,fit_intercept=False)
bayesian.fit(X_new_present,World_cases)
predict_bayesian = bayesian.predict(X_new_future)
test_predictor(bayesian_poly,bayesian,'Bayesian Ridge Predictions',False)

plot_predictions(World_cases, predict_bayesian, adjusted_dates, future_forcast, 'magenta','Bayesian Ridge Predictions')

Show_prediction_by_day('Bayesian Ridge predictions',predict_bayesian)

"""###Predictionns of SVM for Greece"""

Greece_cases = Cumulative_cases[81]
Greece_cases = np.array(Greece_cases).reshape(-1, 1)
SVΜ_cases = SVR(shrinking=True, kernel="poly", gamma=0.01, epsilon=0.5, degree=4, C=0.05, coef0=3.0)
Greece_cases.shape = (len(Greece_cases),)
SVΜ_cases.fit(Passed_days_list,Greece_cases)
predict_SVΜ_Gr = SVΜ_cases.predict(future_forcast) 
plot_predictions(Greece_cases, predict_SVΜ_Gr, adjusted_dates, future_forcast, 'red','SVM Predictions for Greece')

Show_prediction_by_day('SVM predictions for Greece',predict_SVΜ_Gr)

poly_reg = PolynomialFeatures(degree=10,interaction_only=True)
lin_reg = LinearRegression()
lin_reg.fit(X_poly_present, Greece_cases)
predict_polynomialR_gr = lin_reg.predict(X_poly_future)
plot_predictions(Greece_cases, predict_polynomialR_gr, adjusted_dates, future_forcast, 'orange','Polynomial Regression Predictions for Greece')

Show_prediction_by_day('Polynomial Regression predictions for Greece',predict_polynomialR_gr)

bayesian_poly = PolynomialFeatures(degree=5)
bayesian = BayesianRidge(n_iter=1000,compute_score=True,fit_intercept=False,tol=0.0001)
bayesian.fit(X_new_present,Greece_cases)
predict_bayesian_gr = bayesian.predict(X_new_future)
plot_predictions(Greece_cases, predict_bayesian_gr, adjusted_dates, future_forcast, 'magenta','Bayesian Ridge Predictions in Greece')

Show_prediction_by_day(' Bayesian Interpolation predictions for Greece',predict_bayesian_gr)

"""###Prediction using the Prophet model"""

World_cases = np.array(World_cases)
World_cases.shape = (len(World_cases),1)
print(World_cases.shape)
Dates_reported = np.array(Dates_reported)
Dates_reported.shape = (len(Dates_reported),1)
print(Dates_reported.shape)
World_cases_perday = np.concatenate((Dates_reported,World_cases), axis=1)
print((World_cases_perday.shape))
World_cases_perday = pd.DataFrame(data=World_cases_perday)
World_cases_perday.columns = ['ds', 'y']
print(World_cases_perday)

ph = Prophet(n_changepoints=31, changepoint_prior_scale=0.999 ,interval_width=0.999,daily_seasonality=True, seasonality_mode='additive',seasonality_prior_scale=10)
ph.fit(World_cases_perday)
World_cases_perday.tail()

future_prediction = ph.make_future_dataframe(periods=10)
future_prediction.tail()

forecast = ph.predict(future_prediction)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)

y = np.array(forecast['yhat'])[-10:]
y_l = np.array(forecast['yhat_lower'])[-10:]
y_u = np.array(forecast['yhat_upper'])[-10:]
for i in range(len(y)):
  y[i] = int (y[i])
  y_l[i] = int (y_l[i])
  y_u[i] = int (y_u[i])
print(y)
print(y_l)
print(y_u)

future_plot = ph.plot(forecast, figsize=(14,8), xlabel='days', ylabel='cases')

trends_weekly_dayly_plot = ph.plot_components(forecast, figsize=(14,8))

"""###Map Visualization"""

import collections

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
countries_geo = f'{url}/world-countries.json'
print(countries_geo)
with urllib.request.urlopen(countries_geo) as f:
    html = f.read().decode('utf-8')

my_dictionary = json.loads(html)

denominations_json = []
for index in range(len(my_dictionary['features'])):
    denominations_json.append(my_dictionary['features'][index]['properties']['name'])
    
denominations_json = sorted(denominations_json)
#print((denominations_json))

Overall = pd.DataFrame(data=Overall_per_country)
#columns_titles = [0,1]
#Overall=Overall.reindex(columns=columns_titles)
dataframe_names = Overall[0].tolist()
print(len(dataframe_names))
Overall.replace(dict (zip(dataframe_names, denominations_json)), inplace=True)

def make_right(index,id):
  for k in range(1,6):
    Overall[k][index] = Overall_per_country[id][k]

for i in range(len(Overall)):
  for j in Overall_per_country:
    x = similar(Overall[0][i],j[0])
    if (x>=0.99):
      Overall[1][i] = j[1]
      Overall[2][i] = j[2]
      Overall[3][i] = j[3]
      Overall[4][i] = j[4]
      Overall[5][i] = j[5]
  if (Overall[0][i]=="Russia"):
    make_right(i,175)
  if (Overall[0][i]== "Iran"):
    make_right(i,98)

#print(Overall)
for i in range(len(Overall)):
  print(Overall[0][i], Overall[1][i], "           ", Overall_per_country[i][0], Overall_per_country[i][1])

bins = list (Overall[1].quantile([0, 0.8, 0.93, 0.96, 0.98, 0.987, 0.991,1]))
Legends = ['Cases','Deaths','Mortality', 'Cases per milion', 'Deaths per milion']
m = folium.Map(location=[10,-10],width='80%',height='80%',top='0%',left='0%',position='relative',tiles='OpenStreetMap',max_zoom=20, min_zoom=0,zoom_start=2)
folium.Choropleth(
    geo_data=countries_geo,
    name='choropleth',
    data=Overall,
    columns=[0, 1],
    key_on='properties.name',
    fill_color='OrRd',
    fill_opacity=0.9,
    line_opacity=0.6,
    legend_name=Legends[0],
    bins=bins,
    smooth_factor = 1,
    reset=True
).add_to(m)
folium.LayerControl().add_to(m)
m

bins = list (Overall[2].quantile([0, 0.8, 0.91, 0.93, 0.96, 0.98, 0.991,1]))
Legends = ['Cases','Deaths','Mortality', 'Cases per milion', 'Deaths per milion']
m = folium.Map(location=[10,-10],width='80%',height='80%',top='0%',left='0%',position='relative',tiles='OpenStreetMap',max_zoom=20, min_zoom=0,zoom_start=2)
folium.Choropleth(
    geo_data=countries_geo,
    name='choropleth',
    data=Overall,
    columns=[0, 2],
    key_on='properties.name',
    fill_color='BuPu',
    fill_opacity=0.9,
    line_opacity=0.6,
    legend_name=Legends[1],
    bins=bins,
    smooth_factor = 1,
    reset=True
).add_to(m)
folium.LayerControl().add_to(m)
m

