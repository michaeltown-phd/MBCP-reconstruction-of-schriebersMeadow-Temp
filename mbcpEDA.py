#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:50:19 2021

@author: michaeltown
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import os
import datetime as dt

import pickle


# season filter
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
    

# histogram figure function
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

# some file management of the MBCP data

## moving files to data repository,
fileLoc = '/home/michaeltown/work/projects/MBCP/data';
os.chdir(fileLoc);

# copy files to working directory
os.system('cp MBCP2019-2020/MBCP2019-2020_iButton1.csv MBCPcat/');
os.system('cp MBCP2018-2019/MBCP2018-2019_iButton1.csv MBCPcat/');
os.system('cp MBCP2020-2021/MBCP2020-2021_iButton1.csv MBCPcat/');

os.chdir('./MBCPcat/');

fileList = os.listdir('./');
fileList.sort();

# removes the headerline from all but the first file MBCP
os.system('sed -i "1,14d" '+fileList[0]);

for fl in fileList[1:]:
    os.system('sed -i "1,15d" '+fl);

os.system('cat MBCP2018-2019_iButton1.csv MBCP2019-2020_iButton1.csv MBCP2020-2021_iButton1.csv >> MBCP2018-2021_iButton1.csv')



# iButton1 exists in all files
mbcp2018_2021DF = pd.read_csv('MBCP2018-2021_iButton1.csv')

# reformatting the data frame
mbcp2018_2021DF.columns = mbcp2018_2021DF.columns.str.replace(' ','').str.lower().str.replace('/','');
mbcp2018_2021DF.drop(columns = ['unit'],inplace = True);
mbcp2018_2021DF['datetime'] = pd.to_datetime(mbcp2018_2021DF['datetime']);
mbcp2018_2021DF.rename(columns = {'value': 'Tc'},inplace = True);
mbcp2018_2021DF['Tc'] = pd.to_numeric(mbcp2018_2021DF['Tc']);


# process the data in a basic way

mbcp2018_2021DF['date'] = pd.to_datetime(mbcp2018_2021DF.datetime.dt.date);
mbcp2018_2021DF['month'] = mbcp2018_2021DF.datetime.dt.month;
mbcp2018_2021DF['year'] = mbcp2018_2021DF.datetime.dt.year;
mbcp2018_2021DF['season'] = mbcp2018_2021DF.month.apply(seasonFilter);

figureFileLoc = '/home/michaeltown/work/metis/modules/linearRegressionWebScraping/figures/mbcpFigures';
myHistFunc(mbcp2018_2021DF.Tc,range(-50,50,1),-50,50,0,0.15,'Temperature for Schriebers Meadow, 3460 ft, Mt. Baker (201807-202107)','Temperature ($^\circ$C)'
           ,'Fraction',figureFileLoc ,'Tc_iButton01MtBakerhist_201807-202107');


months = range(1,13,1);
monthListH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
monthList = ['01','02','03','04','05','06','07','08','09','10','11','12'];
monDictH = dict(zip(months,monthListH));
monDict = dict(zip(months,monthList));
yearsMBCP = range(2018,2022,1);



for y in yearsMBCP:
    
    for m in months:
        myHistFunc(mbcp2018_2021DF[(mbcp2018_2021DF.year==y)&(mbcp2018_2021DF.month == m)].Tc,range(-50,50,1),-50,50,0,0.15,'T ($^\circ$C) for Schriebers Meadow ('+monDictH[m]+', '+str(y)+') , 3460 ft, Mt. Baker','Temperature ($^\circ$C)'
           ,'Fraction',figureFileLoc,'Tc_iButton01MtBakerhist_'+str(y)+monDict[m]);
        



# save the data frame in a pickle file
mbcp2018_2021DF.to_pickle("/home/michaeltown/work/projects/MBCP/data/MBCPcat/mbcpDF2018-2021.pkl")
