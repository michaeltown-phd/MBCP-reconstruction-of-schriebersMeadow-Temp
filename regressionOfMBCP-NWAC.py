#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:25:20 2021

@author: michaeltown
"""

# Regression of mbcp and NWAC data 
# assumes that mbcp and NWAC data frames exist
# get these data frames from mbcpEDA.py and nwacEDA_HM_PD.py respectively

# initial pass at OLS of tempererature from pan dome to heather meadows


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import scipy.stats as stats

# filters to create categorical columns
def seasonFilter(x):
    

    if (x > 11) | (x < 3):
        return 'djf';
    elif (x >2) & (x <6 ):
        return 'mam';
    elif (x >5) & (x < 9):
        return 'jja';
    elif (x >8) & (x <12 ):
        return 'son';
    else:
        print('I do not recognize that month value. Returning NaN.');
        return np.nan;

# functions necessary to properly average winds (wdir in particular) - could not get these to work - any ideas?

def computeU(ws, wdir):
    
    u = ws*np.sin(wdir*2*np.pi/360)

    return u

def computeV(ws, wdir):
    
    v = ws*np.cos(wdir*2*np.pi/360)

    return v


def computeWs(u,v):
    
    ws = np.sqrt(u**2+v**2);
    
    return ws

def computeWdir(u,v):
    
    wdir = 360*np.arctan(u/v)/(2*np.pi);
    
    return wdir

# intensity reference https://www.baranidesign.com/faq-articles/2020/1/19/rain-rate-intensity-classification

def pFilter(p):
    
    if p == 0:
        return 'no rain';
    elif (p > 0) & (p <= 1):
        return 'drizzle';           # zone 2 is Southerly / Southwesterly flow
    elif (p > 1) & (p <= 2.5):
        return 'light rain';           # zone 3 is westerly flow
    elif (p > 2.5) & (p <= 7.5):
        return 'moderate rain';           # zone 3 is westerly flow
    elif (p > 7.5) :
        return 'heavy rain';           # zone 3 is westerly flow
    else:
        print('I do not recognize that precip value ' + str(p) + '. Returning nan.')
        return np.nan;

def wdirFilter(wdir):
    

    if wdir < 150:
        return 'N-NE';           # zone 1 is Northerly, Northeasterly flow
                            # is probably related to zone 4, but keeping separate
    elif (wdir >= 150) & (wdir < 220 ):
        return 'S';           # zone 2 is Southerly / Southwesterly flow
    elif (wdir >= 220) & (wdir < 300):
        return 'W';           # zone 3 is westerly flow
    elif wdir >= 300:
        return 'N-NW';           # zone 4 is Northerly/Northwesterly flow
    else:
        print('I do not recognize that wind direction ' + str(wdir) + '. Returning nan.')
        return np.nan;


# figure to plot residual plots

def plotResid(yTrue,yResid,xlabelR,ylabelR,titleR):
    
    fig = plt.figure();
    zeroLine = yTrue*0;
    plt.plot(yTrue,yResid,color = 'blue',marker = 'o',alpha = 0.5,ls = 'None');
    plt.plot(yTrue,zeroLine,'k--')
    plt.xlabel(xlabelR);
    plt.ylabel(ylabelR);
    plt.grid;
    plt.title(titleR);
    plt.show;
    

# analysis pipeline to test the improvementment of the model 


#def analysisPipeline(x,y,targetName,testSize):
'''
    xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);


    modelT = sm.OLS(yTrain,xTrain);
    resultsT = modelT.fit();
    
    
    yNewTest = resultsT.predict(xTest);
    residTest = yTest - yNew;
    
    slopes = resultsT.params;
    intercept = resultsT.params.const;
    
    x = np.arange(-20,30);
    x2 = np.arange(-15,25);
    # plot a figure

    fig1 = plt.figure();
    plt.plot(s1.Tc_x,s1.Tc_y,'.',color = 'blue',alpha = 0.5);
    plt.plot(x,x,'--',color = 'black',alpha = 0.3)
    plt.plot(x2,slope*x2+intercept,'--',color = 'red')
    plt.title('Regression of Pan Dome against Schriebers Meadow, 2018-2021');
    plt.ylabel('T (oC) SM');
    plt.xlabel('T (oC) PD');
    plt.text(-20,25,'T_SM = ' + str(np.round(slope,2)) + ' T_PD + ' + str(np.round(intercept,2)));    
    plt.grid()
    plt.legend(['Temperature','1:1 line','regression fit'])
'''

# load the data webscraped (pdDF, hmDF) and collected from the mtns by Town's students.
pdDF = pd.read_pickle("/home/michaeltown/work/metis/modules/linearRegressionWebScraping/data/panDomeDF2014-2021.pkl")
hmDF = pd.read_pickle("/home/michaeltown/work/metis/modules/linearRegressionWebScraping/data/heatherMeadowsDF2014-2021.pkl")
mbcp2018_2021DF = pd.read_pickle("/home/michaeltown/work/projects/MBCP/data/MBCPcat/mbcpDF2018-2021.pkl")



# clean data in the other script later
pdDF.dropna(axis = 0, inplace = True, subset = ['Tc'])

# drop columns not necessary for this analysis
# could keep the min/max windspeeds later in the Pan dome df

dropColpdDF = ['batteryvoltagev','precipitation','totalsnowdepth','barometricpressuremb','equipmenttemperaturedegf',
               'solarradiationwm2','netsolarmjm2','24hoursnow','intermittentshotsnow',
               'soilmoistureavwc','soilmoisturebvwc','soilmoisturecvwc','soiltemperatureadegf',
               'soiltemperaturebdegf','soiltemperaturecdegf','windspeedminimummph',
               'windspeedmaximummph','relativehumidity'];

dropColhmDF = ['batteryvoltagev','totalsnowdepth','barometricpressuremb','equipmenttemperaturedegf',
               'solarradiationwm2','24hoursnow','intermittentshotsnow','netsolarmjm2',
               'soilmoistureavwc','soilmoisturebvwc','soilmoisturecvwc','soiltemperatureadegf',
               'soiltemperaturebdegf','soiltemperaturecdegf','windspeedminimummph',
               'windspeedmaximummph','winddirectiondeg','windspeedaveragemph'];

pdDF.drop(columns = dropColpdDF,inplace = True)
hmDF.drop(columns = dropColhmDF,inplace = True)

# need dummy variables
pdDF['u'] = pdDF.Tc*0;
pdDF['v'] = pdDF.Tc*0;
pdDF['u'] = pdDF.apply(lambda x : -1*pdDF['wsAvgMs']*np.sin(pdDF['winddirectiondeg']*2*np.pi/360))
pdDF['v'] = pdDF.apply(lambda x : -1*pdDF['wsAvgMs']*np.cos(pdDF['winddirectiondeg']*2*np.pi/360))



# need to do some daily averaging to pan dome data
pdDailyDF= pd.DataFrame();
columnsToAvg = ['Tc','u','v']
pdDailyDF = pdDF.groupby(['date'])[columnsToAvg].mean()
pdDailyDF['month'] = pdDailyDF.index.month
pdDailyDF['season'] = pdDailyDF.month.apply(seasonFilter);
pdDailyDF['wsAvgMs'] = pdDailyDF['u']*0; # dummy column
pdDailyDF['wdir'] = pdDailyDF['u']*0; # dummy column
pdDailyDF['wsAvgMs'] = pdDailyDF.apply(lambda x : np.sqrt(pdDailyDF['u']**2+pdDailyDF['v']**2));
pdDailyDF['wdir'] = pdDailyDF.apply(lambda x : 180*(np.arctan2(-1*pdDailyDF.u,-1*pdDailyDF.v)/np.pi)+180)
pdDailyDF['wdirCat'] = pdDailyDF['wdir'].apply(wdirFilter)

# need to do some daily averaging to pan dome data
hmDailyDF= pd.DataFrame();
columnsToAvg = ['Tc','precipitation']
hmDailyDF = hmDF.groupby(['date'])[columnsToAvg].mean()
hmDailyDF['pCat'] = hmDailyDF['precipitation'].apply(pFilter)


# some nans produced in the wdir calc
pdDailyDF.dropna(axis = 0, inplace = True, subset = ['wdir'])
hmDailyDF.dropna(axis = 0, inplace = True, subset = ['precipitation'])

pdDailyDF = pd.concat((pdDailyDF,pd.get_dummies(pdDailyDF['season'], drop_first=True)), axis=1)
pdDailyDF = pd.concat((pdDailyDF,pd.get_dummies(pdDailyDF['wdirCat'], drop_first=True)), axis=1)
hmDailyDF = pd.concat((hmDailyDF,pd.get_dummies(hmDailyDF['pCat'], drop_first=True)), axis=1) # base case is drizzle


mbcp2018_2021DF['date'] = pd.to_datetime(mbcp2018_2021DF['datetime'].dt.date)
mbcpTmean = mbcp2018_2021DF.groupby(['date'])['Tc'].mean();

# function inputs
suffixesMerge = ['_pd','_ss'];
# suffixesMerge2 = ['_hm',''];
columns = ['Tc_pd']; 
#columns = ['Tc_pd','wsAvgMs']; 
#columns = ['Tc_pd','wsAvgMs','N-NW','S','W']; 
# columns = ['Tc_pd','wsAvgMs','jja','mam','son']; 
# columns = ['Tc_pd','Tc','wsAvgMs','N-NW','S','W','jja','mam','son','heavy rain','light rain','moderate rain','no rain']; 
# columns = ['Tc','wsAvgMs','N-NW','S','W','jja','mam','son','heavy rain','light rain','moderate rain','no rain']; 
# columns = ['Tc','N-NW','S','W','jja','mam','son','heavy rain','light rain','moderate rain','no rain']; 
# columns = ['Tc','jja','mam','son','heavy rain','light rain','moderate rain','no rain']; 



testSize = 0.3;



s1 = pd.merge(pdDailyDF, mbcpTmean, how='inner', on=['date'],suffixes = suffixesMerge)
s1 = pd.merge(s1, hmDailyDF, how='inner', on=['date'])


# column management

# snspairplot to see what is happening
# sns.pairplot(s1)

x = s1[columns];
x = sm.add_constant(x);
y = s1.Tc_ss;



# train/test code
xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);



modelT = LinearRegression();
resultsT = modelT.fit(xTrain,yTrain);

# zscore modeling
modelTOLS_allZ = sm.OLS(stats.zscore(yTrain),sm.add_constant(xTrain.iloc[:,1:].apply(stats.zscore)));
resultsTOLS_allZ = modelTOLS_allZ.fit();

modelTOLS_all = sm.OLS(yTrain,xTrain);
resultsTOLS_all = modelTOLS_all.fit();


yNewTest = resultsT.predict(xTest);
residTest = yTest - yNewTest;
plotResid(yTest, residTest, 'T (oC)', 'ylabelRT (oC)', 'Residual Test of 2018-2021 for S. Mt. Baker - schreibers meadow')

yNewTestOLS_all = resultsTOLS_all.predict(xTest);
residTestOLS_all = yTest - yNewTest;
plotResid(yTest, residTestOLS_all, 'T (oC)', 'ylabelRT (oC)', 'Residual Test of 2018-2021 for S. Mt. Baker - schreibers meadow')


# looking at the training data 
figQQ = sm.qqplot(yTrain,line = '45');

# z-transformation to see how this affects normality
figQQ = sm.qqplot(stats.zscore(yTrain),line = '45');
plt.title('zscore transform hm temp data')

slopes = resultsTOLS.params;
intercept = resultsTOLS.params.const;

print(resultsTOLS.summary())


## cross validation scores
testScore = cross_val_score(modelT, np.asarray(yTest).reshape(-1, 1), np.asarray(yNewTest).reshape(-1, 1))
print('mean test score = ' + str(np.round(np.mean(testScore),3)))


# plot the different seasons as color-coded scatter plot
s1jja = s1.where(s1.jja == 1)
s1mam = s1.where(s1.mam == 1)
s1son = s1.where(s1.son == 1)
s1djf = s1.where((s1.mam == 0)&(s1.son == 0)&(s1.jja == 0))

## quick test of the seasonal results regression
## looking only at mam results
s1mamTest = s1mam.dropna(axis = 0, subset = ['mam'])
x = s1mamTest['Tc'];
x = sm.add_constant(x);
y = s1mamTest.Tc_ss;

# train/test code
xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);
modelTOLS = sm.OLS(yTrain,xTrain);
resultsTOLS = modelTOLS.fit();
print('mam results')
print(resultsTOLS.summary())
Tcoef_mam = resultsTOLS.params.Tc;
Tconst_mam = resultsTOLS.params.const;


## looking only at jja results
s1jjaTest = s1jja.dropna(axis = 0, subset = ['jja'])
x = s1jjaTest['Tc'];
x = sm.add_constant(x);
y = s1jjaTest.Tc_ss;

# train/test code
xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);
modelTOLS = sm.OLS(yTrain,xTrain);
resultsTOLS = modelTOLS.fit();
print('jja results')
print(resultsTOLS.summary())
Tcoef_jja = resultsTOLS.params.Tc;
Tconst_jja = resultsTOLS.params.const;


## looking only at son results
s1sonTest = s1son.dropna(axis = 0, subset = ['son'])
x = s1sonTest['Tc'];
x = sm.add_constant(x);
y = s1sonTest.Tc_ss;

# train/test code
xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);
modelTOLS = sm.OLS(yTrain,xTrain);
resultsTOLS = modelTOLS.fit();
print('son results')
print(resultsTOLS.summary())
Tcoef_son = resultsTOLS.params.Tc;
Tconst_son = resultsTOLS.params.const;



## looking only at dfj results
s1djfTest = s1djf.dropna(axis = 0)
x = s1djfTest['Tc'];
x = sm.add_constant(x);
y = s1djfTest.Tc_ss;

# test code
xTrain, xTest, yTrain, yTest = train_test_split(x,y,random_state = 42,train_size = testSize);
modelTOLS = sm.OLS(yTrain,xTrain);
resultsTOLS = modelTOLS.fit();
print('djf results')
print(resultsTOLS.summary())
Tcoef_djf = resultsTOLS.params.Tc;
Tconst_djf = resultsTOLS.params.const;


# test code
print('ztransform results')
print(resultsTOLS_allz.summary())
Tcoef_pdz = resultsTOLS_allz.params.Tc;
Tconst_pdz = resultsTOLS_allz.params.const;


fig = plt.figure;
plt.plot(s1djf.Tc,s1djf.Tc_ss,'ob',alpha = 0.5,ls = 'None')
plt.plot(s1mam.Tc,s1mam.Tc_ss,'g',alpha = 0.5,marker = '<',ls = 'None')
plt.plot(s1jja.Tc,s1jja.Tc_ss,'r',alpha = 0.5,marker = '>',ls = 'None')
plt.plot(s1son.Tc,s1son.Tc_ss,color='orange',alpha = 0.5,marker = '*',ls = 'None')
xVal = np.arange(np.min(s1djf.Tc),np.max(s1djf.Tc))
plt.plot(xVal,Tcoef_djf*xVal+Tconst_djf,color = 'darkblue',ls='--',linewidth=2.5)
xVal = np.arange(np.min(s1mam.Tc),np.max(s1mam.Tc))
plt.plot(xVal,Tcoef_mam*xVal+Tconst_mam,color = 'darkgreen',ls='--',linewidth=2.5)
xVal = np.arange(np.min(s1jja.Tc),np.max(s1jja.Tc))
plt.plot(xVal,Tcoef_jja*xVal+Tconst_jja,color = 'darkred',ls='--',linewidth=2.5)
xVal = np.arange(np.min(s1son.Tc),np.max(s1son.Tc))
plt.plot(xVal,Tcoef_son*xVal+Tconst_son,color = 'chocolate',ls='--',linewidth=2.5)
plt.xlim([-20,30])
plt.ylim([-20,30])
plt.plot([-20,30],[-20,30],'k--',alpha = 0.5)
plt.grid();
plt.legend(['djf','mam','jja','son'])
plt.xlabel('Heather Meadows, Mt. Baker T ($^\circ$C)')
plt.ylabel('Schriebers Meadow T ($^\circ$C)')


## subplots
fig = plt.figure();
plt.subplot(221);
plt.plot(s1djf.Tc,s1djf.Tc_ss,'ob',alpha = 0.5,ls = 'None')
xVal = np.arange(np.min(s1djf.Tc),np.max(s1djf.Tc))
plt.plot(xVal,Tcoef_djf*xVal+Tconst_djf,color = 'black',ls='--',linewidth=2.5)
plt.xlim([-20,30])
plt.ylim([-20,30])
plt.plot([-20,30],[-20,30],'k--',alpha = 0.5)
plt.grid();
plt.legend(['djf'])
plt.ylabel('Sch Mead, T ($^\circ$C)')

plt.subplot(222);
plt.plot(s1mam.Tc,s1mam.Tc_ss,'g',alpha = 0.5,marker = '<',ls = 'None')
xVal = np.arange(np.min(s1mam.Tc),np.max(s1mam.Tc))
plt.plot(xVal,Tcoef_mam*xVal+Tconst_mam,color = 'black',ls='--',linewidth=2.5)
plt.xlim([-20,30])
plt.ylim([-20,30])
plt.plot([-20,30],[-20,30],'k--',alpha = 0.5)
plt.grid();
plt.legend(['mam'])
plt.xlabel('Heath Mead, T ($^\circ$C)')

plt.subplot(223);
plt.plot(s1jja.Tc,s1jja.Tc_ss,'r',alpha = 0.5,marker = '>',ls = 'None')
xVal = np.arange(np.min(s1jja.Tc),np.max(s1jja.Tc))
plt.plot(xVal,Tcoef_jja*xVal+Tconst_jja,color = 'black',ls='--',linewidth=2.5)
plt.ylim([-20,30])
plt.plot([-20,30],[-20,30],'k--',alpha = 0.5)
plt.grid();
plt.legend(['jja'])
plt.ylabel('Sch Mead, T ($^\circ$C)')
plt.xlabel('Heath Mead, T ($^\circ$C)')

plt.subplot(224);
plt.plot(s1son.Tc,s1son.Tc_ss,color='orange',alpha = 0.5,marker = '*',ls = 'None')
xVal = np.arange(np.min(s1son.Tc),np.max(s1son.Tc))
plt.plot(xVal,Tcoef_son*xVal+Tconst_son,color = 'black',ls='--',linewidth=2.5)
plt.xlim([-20,30])
plt.ylim([-20,30])
plt.plot([-20,30],[-20,30],'k--',alpha = 0.5)
plt.grid();
plt.legend(['son'])
plt.xlabel('Heath Mead, T ($^\circ$C)')


# temperature reconstruction for 2014-2021

# create a longer data frame will all the same elements.

s12014_2021 = pd.merge(pdDailyDF, hmDailyDF, how='inner', on=['date'])
columns = ['Tc_y','jja','mam','son','heavy rain','light rain','moderate rain','no rain']; 
x = s12014_2021[columns];
x = sm.add_constant(x);

Tc2014_2021 = resultsTOLS_all.predict(x)

fig3 = plt.figure();
plt.plot(s12014_2021.index,Tc2014_2021,label = 'T reconst');
plt.plot(mbcp2018_2021DF.date,mbcp2018_2021DF.Tc,'k',alpha = 0.5, label = 'T meas')
plt.grid()
plt.ylim([-20,40])
plt.legend();
plt.title('South Side of Mt. Baker temperature, measured and reconstructed');
plt.xlabel('Date');
plt.ylabel('T ($^\circ$C)')


s12014_2021 = pdDailyDF.Tc
columns = ['Tc']; 
x = s12014_2021;
x = sm.add_constant(x);

Tc2014_2021 = resultsTOLS_allZ.predict(x)
Tc2014_2021 = Tc2014_2021*np.std(pdDailyDF.Tc)+np.mean(pdDailyDF.Tc)


fig3 = plt.figure();
plt.plot(s12014_2021.index,Tc2014_2021,label = 'T reconst');
plt.plot(mbcp2018_2021DF.date,mbcp2018_2021DF.Tc,'k',alpha = 0.5, label = 'T meas')
plt.grid()
plt.ylim([-20,40])
plt.legend();
plt.title('South Side of Mt. Baker temperature, measured and reconstructed');
plt.xlabel('Date');
plt.ylabel('T ($^\circ$C)')


'''
MVP Code 

x = s1.Tc_x;
x = sm.add_constant(x);
y = s1.Tc_y;

modelT = sm.OLS(y,x);
resultsT = modelT.fit();


mbcp2018_2021DF['T_predict'] = resultsT.predict(x);
mbcp2018_2021DF['resid'] = mbcp2018_2021DF['Tc'] - mbcp2018_2021DF['T_predict'];

slope = resultsT.params.Tc_x;
intercept = resultsT.params.const;

x = np.arange(-20,30);
x2 = np.arange(-15,25);
# plot a figure
fig1 = plt.figure();
plt.plot(s1.Tc_x,s1.Tc_y,'.',color = 'blue',alpha = 0.5);
plt.plot(x,x,'--',color = 'black',alpha = 0.3)
plt.plot(x2,slope*x2+intercept,'--',color = 'red')
plt.title('Regression of Pan Dome against Schriebers Meadow, 2018-2021');
plt.ylabel('T (oC) SM');
plt.xlabel('T (oC) PD');
plt.text(-20,25,'T_SM = ' + str(np.round(slope,2)) + ' T_PD + ' + str(np.round(intercept,2)));    
plt.grid()
plt.legend(['Temperature','1:1 line','regression fit'])
'''

'''
Other code to use, maybe
modelTDx1x2 = sm.OLS(trainingData.y,Xdata);
resultsX1X2 = modelTDx1x2.fit();
print(resultsX1X2.summary());
'''