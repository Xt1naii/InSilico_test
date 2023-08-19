import pandas as pd
import requests
import csv
import logging
import sys
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor

def remove_words(x: str):
    x = x.replace('antagonists', '')
    x = x.replace('inhibitors', '')
    x = x.replace('activators', '')
    x = x.replace('antagonist', '')
    x = x.replace('inhibitor', '')
    x = x.replace('activator', '')
    x = x.replace('stimulants', '')
    x = x.replace('stimulant', '')
    x = x.replace('agonists', '')
    x = x.replace('agonist', '')
    x = x.replace('replacements', '')
    x = x.replace('modulators', '')
    return x.strip()

def prepare_dataset(file):
   df = pd.read_parquet(file)
   df.set_index('name')
   df[['events1', 'events2', 'events3']] = pd.DataFrame(df.recent_events.fillna(' ').to_list())
   mechanism = pd.DataFrame(df.mechanism.fillna(' ').to_list())
   df = df[~mechanism.eq("Undefined mechanism").any(axis=1)]
   df = df[~mechanism.eq("").any(axis=1)]
   
   disc2_1 = df[df['events1'].str.contains("Discontinued-II|Discontinued - Phase-II")==True]
   disc2_2 = df[df['events2'].str.contains("Discontinued-II|Discontinued - Phase-II")==True]
   disc2_3 = df[df['events3'].str.contains("Discontinued-II|Discontinued - Phase-II")==True]
   
   disc_2_all = pd.concat([disc2_1, disc2_2, disc2_3], axis=0)
   
   disc3_1 = df[df['events1'].str.contains("Discontinued-III|Discontinued - Phase-III")==True]
   disc3_2 = df[df['events2'].str.contains("Discontinued-III|Discontinued - Phase-III")==True]
   disc3_3 = df[df['events3'].str.contains("Discontinued-III|Discontinued - Phase-III")==True]
   
   disc_3_all = pd.concat([disc3_1, disc3_2, disc3_3], axis=0)
   
   marketed = df[df.dev_phase == 'Marketed']
   
   passed_to_3 = pd.concat([disc_3_all, marketed], axis=0)
   passed_to_3['passed_to_3'] = 1
   disc_2_all['passed_to_3'] = 0
   
   data = pd.concat([passed_to_3, disc_2_all], axis=0)
   data = data.reset_index(drop=True)

   drug_class = pd.DataFrame(data['class'].fillna(' ').to_list())
   data = data[~drug_class.eq("Antibacterials").any(axis=1)]
   data = data[~drug_class.eq("Antiretrovirals").any(axis=1)]
   data = data[~drug_class.eq("Antivirals").any(axis=1)]
   data = data[~drug_class.eq("Antisense oligonucleotides").any(axis=1)]
   data = data[~drug_class.eq("Bacterial vaccines").any(axis=1)]
   data = data[~drug_class.eq("Viral vaccines").any(axis=1)]
   data = data[~drug_class.eq("Radioisotopes").any(axis=1)]
   data = data[~drug_class.eq("Electrolytes").any(axis=1)]
   data = data[~drug_class.eq("Antiprotozoals").any(axis=1)]
   data = data[~drug_class.eq("Diagnostic agents").any(axis=1)]
   data = data[~drug_class.eq("Ionising radiation emitters").any(axis=1)]
   data = data[~drug_class.eq("Positron-emission tomography enhancers").any(axis=1)]
   data = data[~drug_class.eq("Radiography enhancers").any(axis=1)]
   return data
