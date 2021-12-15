#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:09:11 2021

@author: michaeltown
"""
from bs4 import BeautifulSoup
import requests
import time, os



from selenium import webdriver
from selenium.webdriver.common.keys import Keys

### open table practice
fp=webdriver.FirefoxProfile()
fp.set_preference("browser.helperApps.neverAsk.saveToDisk","text/csv")
driver = webdriver.Firefox(fp)
driver.get('https://nwac.us/data-portal/location/mt-baker-ski-area/')
time.sleep(1)  #pause to be sure page has loaded

# eventually want to loop around the length of these drop down menus
# use beautifulsoup to find the length of these menus for each menu

# for Heather Meadows - no need to change the data logger id

lenYearMenu = 8;
yearSelect = driver.find_elements_by_xpath('//select[@name="year"]')

for i in range(lenYearMenu):
    dataDownload = driver.find_element_by_xpath('//input[@value="Get CSV file"]')
    dataDownload.click()
    
    if i < lenYearMenu-1:
        yearSelect[1].send_keys(Keys.DOWN) # walk down the menu each time the loop executes, don't do on last time around

    
# toggles the field down one element to get pan dome data    
driver.find_element_by_name("datalogger_id").send_keys(Keys.DOWN)

for i in range(lenYearMenu):
    dataDownload = driver.find_element_by_xpath('//input[@value="Get CSV file"]')
    dataDownload.click()
    
    if i < lenYearMenu-1:
        yearSelect[1].send_keys(Keys.UP) # walk UP the menu each time the loop executes, don't do on last time around

driver.quit();


## moving files to data repository,
os.chdir('/home/michaeltown/Downloads/');
fileList = os.listdir('/home/michaeltown/Downloads/');
fileList.sort();

fileListHM = [item for item in fileList if 'Heather' in item]
fileListPD = [item for item in fileList if 'Pan' in item]

# removes the headerline from all but the first file for Heather Meadows
for fl in fileListHM[1:]:
    os.system('sed -i "1d" '+fl);

# removes the headerline from all but the first file for Pan Dome
for fl in fileListPD[1:]:
    os.system('sed -i "1d" '+fl);


for fl in fileList:
    os.rename('/home/michaeltown/Downloads/'+fl,'/home/michaeltown/work/metis/modules/linearRegressionWebScraping/data/'+fl);


# concatenating the like files (ideally using the naming convention drawn from bs) 
os.chdir('/home/michaeltown/work/metis/modules/linearRegressionWebScraping/data/');
os.system('cat *Heather*.csv >> MtBaker-HeatherMeadows2014-2021.csv');
os.system('cat *Pan*.csv >> MtBaker-PanDome2014-2021.csv');

