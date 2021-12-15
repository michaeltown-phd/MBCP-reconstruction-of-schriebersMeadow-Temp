#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:51:16 2021

@author: michaeltown
"""

# NWAC data EDA analysis of the Heather Meadows and Pan Dome data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import datetime as dt
    
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
    
# function to convert degF to degC
def convertFtoC(tf):
    return (tf-32)*5/9;

# function to convert mph to m/s
def convertwsMPHtoMS(ws):    
    return ws*0.44704;

# function to convert in to mm
def convertprecipIntomm(p):    
    return p*25.4;


def myHistFunc(x,ra,xll,xlu,yll,ylu,title,xlab,ylab,fileloc,filename):
    fig1 = plt.figure();
    plt.hist(x,bins = ra,color = 'blue',alpha = 0.5, density = True);
    plt.xlim([xll,xlu]);
    plt.ylim([yll,ylu]);
    plt.title(title);
    plt.ylabel(ylab);
    plt.xlabel(xlab);    
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll,'n = '+str(len(x)));
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll-(ylu-yll)/10,'mean = '+str(np.round(np.mean(x),2)));
    plt.text((xlu-xll)*2/3+xll,(ylu-yll)/2+yll-2*(ylu-yll)/10,'stdev = '+str(np.round(np.std(x),2)));
    os.chdir(fileloc)
    fig1.savefig(filename+'.jpg')

# load the data framces
os.chdir('/home/michaeltown/work/metis/modules/linearRegressionWebScraping/data/');

pdDF = pd.read_csv('MtBaker-PanDome2014-2021.csv');
hmDF = pd.read_csv('MtBaker-HeatherMeadows2014-2021.csv');

# strip all columns of spaces and make all letters lowercase

pdDF.columns = pdDF.columns.str.replace(' ','').str.lower().str.replace('/','').str.replace('(','').str.replace(')','').str.replace(')','').str.replace('%','').str.replace('"','').str.replace('.','');
hmDF.columns = hmDF.columns.str.replace(' ','').str.lower().str.replace('/','').str.replace('(','').str.replace(')','').str.replace(')','').str.replace('%','').str.replace('"','').str.replace('.','');

# convert to pandas date/time 
pdDF.datetimepst = pd.to_datetime(pdDF.datetimepst);
hmDF.datetimepst = pd.to_datetime(hmDF.datetimepst);


# prep the time series for daily, monthly, and seasonal composites - some conversions to SI units
pdDF['date'] = pd.to_datetime(pdDF.datetimepst.dt.date);
pdDF['month'] = pdDF.datetimepst.dt.month;
pdDF['year'] = pdDF.datetimepst.dt.year;
pdDF['season'] = pdDF.month.apply(seasonFilter);
pdDF['Tc'] = pdDF.temperaturedegf.apply(convertFtoC);
pdDF['precipitation'] = pdDF.precipitation.apply(convertprecipIntomm);
pdDF['wsAvgMs'] = pdDF.windspeedaveragemph.apply(convertwsMPHtoMS);
pdDF['wsMinMs'] = pdDF.windspeedminimummph.apply(convertwsMPHtoMS);
pdDF['wsMaxMs'] = pdDF.windspeedmaximummph.apply(convertwsMPHtoMS);

hmDF['date'] = pd.to_datetime(hmDF.datetimepst.dt.date);
hmDF['month'] = hmDF.datetimepst.dt.month;
hmDF['year'] = hmDF.datetimepst.dt.year;
hmDF['season'] = hmDF.month.apply(seasonFilter);
hmDF['Tc'] = hmDF.temperaturedegf.apply(convertFtoC);
hmDF['precipitation'] = hmDF.precipitation.apply(convertprecipIntomm);
hmDF['wsAvgMs'] = hmDF.windspeedaveragemph.apply(convertwsMPHtoMS);
hmDF['wsMinMs'] = hmDF.windspeedminimummph.apply(convertwsMPHtoMS);
hmDF['wsMaxMs'] = hmDF.windspeedmaximummph.apply(convertwsMPHtoMS);


# TODO in this space, please drop all the unnecessary columns


# plot the initial EDA histograms
fileLoc = '/home/michaeltown/work/metis/modules/linearRegressionWebScraping/figures/nwacFigures/';
myHistFunc(pdDF.Tc,range(-50,50,1),-50,50,0,0.15,'Temperature for Pan Dome, 5020 ft, Mt. Baker (2014-2021)','Temperature ($^\circ$C)'
           ,'Fraction',fileLoc,'Tc_panDomeMtBakerhist_2014-2021');

myHistFunc(pdDF.wsAvgMs,range(-2,30,1),-2,30,0,0.3,'Wind Speed for Pan Dome, 5020 ft, Mt. Baker (2014-2021)','Wind Speed (m/s)'
           ,'Fraction',fileLoc,'wsAvg_panDomeMtBakerhist_2014-2021');

myHistFunc(pdDF.winddirectiondeg,range(-0,361,1),-5,365,0,0.01,'WDir for Pan Dome, 5020 ft, Mt. Baker (2014-2021)','Wind Dir (deg)'
           ,'Fraction',fileLoc,'wdir_panDomeMtBakerhist_2014-2021');



# heather meadows
myHistFunc(hmDF.Tc,range(-50,50,1),-50,50,0,0.15,'Temperature for Heather Meadows, 4210 ft, Mt. Baker (2014-2021)','Temperature ($^\circ$C)'
           ,'Fraction',fileLoc,'Tc_heatherMeadowsMtBakerhist_2014-2021');

myHistFunc(hmDF.precipitation,np.arange(0,10,1),0,10,0,1,'Precip for Heather Meadows, 4210 ft, Mt. Baker (2014-2021)','Precip (mm)'
           ,'Fraction',fileLoc,'Precip_heatherMeadowsMtBakerhist_2014-2021');

# 
#myHistFunc(pdDF.wsAvgMs,range(-2,30,1),-2,30,0,0.3,'Wind Speed for Heather Meadows, 4210 ft, Mt. Baker (2014-2021)','Wind Speed (m/s)'
#           ,'Fraction',fileLoc,'wsAvg_heatherMeadowsMtBakerhist_2014-2021');




# yearsNWAC = range(2014,2022,1);
months = range(1,13,1);
monthListH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
monthList = ['01','02','03','04','05','06','07','08','09','10','11','12'];
monDictH = dict(zip(months,monthListH));
monDict = dict(zip(months,monthList));
yearsNWAC = range(2014,2022,1);



for y in yearsNWAC:
    
    for m in months:
        myHistFunc(pdDF[(pdDF.year==y)&(pdDF.month == m)].Tc,range(-50,50,1),-50,50,0,0.15,'T ($^\circ$C) for Pan Dome ('+monDictH[m]+', '+str(y)+') , 5020 ft, Mt. Baker (2014-2021)','Temperature ($^\circ$C)'
           ,'Fraction',fileLoc,'Tc_panDomeMtBakerhist_'+str(y)+monDict[m]);
        
        myHistFunc(pdDF[(pdDF.year==y)&(pdDF.month == m)].wsAvgMs,range(-2,30,1),-2,30,0,0.3,'WS for Pan Dome ('+monDictH[m]+', '+str(y)+'), 5020 ft, Mt. Baker (2014-2021)','Wind Speed (m/s)'
           ,'Fraction',fileLoc,'wsAvg_panDomeMtBakerhist_'+str(y)+monDict[m]);
        
        myHistFunc(pdDF[(pdDF.year==y)&(pdDF.month == m)].winddirectiondeg,range(-0,361,1),-5,365,0,0.10,'WDir for Pan Dome ('+monDictH[m]+', '+str(y)+'), 5020 ft, Mt. Baker (2014-2021)','Wind Dir (deg)'
           ,'Fraction',fileLoc,'wdir_panDomeMtBakerhist_'+str(y)+monDict[m]);

        myHistFunc(hmDF[(pdDF.year==y)&(pdDF.month == m)].Tc,range(-50,50,1),-50,50,0,0.15,'T ($^\circ$C) for Heath Mead ('+monDictH[m]+', '+str(y)+'), 4250 ft, Mt. Baker (2014-2021)','Temperature ($^\circ$C)'
           ,'Fraction',fileLoc,'Tc_heatherMeadowsMtBakerhist_'+str(y)+monDict[m]);

        myHistFunc(hmDF.precipitation[(pdDF.year==y)&(pdDF.month == m)],np.arange(0,10,1),0,10,0,1,'Precip for Heather Meadows, 4210 ft, Mt. Baker ('+monDictH[m]+', '+str(y)+')','Precip (mm)'
           ,'Fraction',fileLoc,'Precip_heatherMeadowsMtBakerhist_'+str(y)+monDict[m]);


# # group by date 
# pdDFdateGmean = pdDF.groupby(['date'])['Tc','relativehumidity','wsAvgMs','winddirectiondeg'].mean();
# pdDFdateGstd = pdDF.groupby(['date'])['Tc','relativehumidity','wsAvgMs'].agg(np.std);
# pdDFdateGmax = pdDF.groupby(['date'])['Tc','relativehumidity','wsAvgMs'].max();
# pdDFdateGmin = pdDF.groupby(['date'])['Tc','relativehumidity','wsAvgMs'].max();

# pdDFdateGmean = pdDFdateGmean.unstack(level= 1);     
# pdDFdateGstd = pdDFdateGstd.unstack(level = 1); 
# pdDFdateGmax = pdDFdateGmax.unstack(level = 1);
# pdDFdateGmin = pdDFdateGmin.unstack(level = 1);


# pickle the data frames

pdDF.to_pickle("../data/panDomeDF2014-2021.pkl")
hmDF.to_pickle("../data/heatherMeadowsDF2014-2021.pkl")