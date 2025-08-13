#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from textwrap import wrap
from tqdm import tqdm
import matplotlib as mpl
from datetime import datetime
import ipaddress

import multiprocessing as mp

from period_detection import find_period

import random
import colorsys
import time

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.expand_frame_repr', False)

import sys
suffix = sys.argv[1]

from ipv6_scanner.config import *

filename = f'telescope-t{suffix}_data.parquet'
df_file = f'{PROCESSED_DATA_DIR}/{filename}'
announcement_log_file = f'{PROCESSED_DATA_DIR}/prefix-announcements.csv'

announcement_df = pd.read_csv(announcement_log_file)

vertical_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in announcement_df[1:].Timestamp_From.unique()]

def unique_value(x):
    u = x.unique()
    if len(u) > 1:
        raise ValueError('Non unique value per group')
    else:
        return u[0]


df = pd.read_parquet(df_file)
print('[*] Load dataframe.')

source_ips_128 = df.scan_source_128.unique()
print('[*] Prepared data 1.')

testing_for_periodicity = df[~(df.is_oneoff_128) & (((df.TCP_Flags=='nan') | (df.TCP_Flags=='··········S·')) & (~df.Source_Address.str.startswith('2001:67c:254:')))]
#testing_for_periodicity = df[~(df.is_oneoff_128)] 

print('[*] Prepared data 2.')

def status_bar(shared_dict,total,starttime):
    while total>len(shared_dict):
        processtime=round(time.time()-starttime,2)
        speed = round(len(shared_dict)/processtime,3)
        if speed==0:
            eta = np.inf
        else:
            eta = round((total-len(shared_dict))/speed,3)
        sys.stdout.write("\r" + f'[*] Status: {len(shared_dict)}/{total} IPs tested in {processtime}s -- {speed} it/s -- ETA: {eta}s')
        sys.stdout.flush()
        time.sleep(3)
    print()
            
def testing_for_periodicity_worker(df,ip,shared_dict,bining='1h',range_freq='h',max_date='2024-07-02'):
    min_date = df.Date.min()
    daterange = pd.date_range(min_date, max_date,freq='h')
    # to few datapoints, no periodicity derivable
    if len(daterange) < 300:
        shared_dict[ip] = -3
        return 1

    tmp = df.groupby('Timestamp').size().resample('1h').count()
    tmp = tmp.reindex(daterange,fill_value=0)
    tmp = tmp.reset_index(name='Value').rename(columns={'index':'date','Value':'value'})
    res_period,res_model,res_criteria = find_period(tmp,path_is_df=True,noprint=True,output_flag=0)
    shared_dict[ip] = res_period
    return 1
    
with mp.Manager() as manager:
    print('[*] Started mp.Manager().')
    shared_dict = manager.dict()
    pool = mp.Pool(processes=mp.cpu_count())
    print('[*] Init process pool.')
    groups = testing_for_periodicity.groupby('scan_source_128',observed=True)
    print('[*] Created groups.')
    start = time.time()
    args_list = [(group, ip, shared_dict) for ip, group in groups]
    end=time.time()
    print(f'[*] Prepared data for periodicity testing. Took {round(end-start,2)}s')
    status_process = mp.Process(target=status_bar, args=(shared_dict,len(args_list),time.time()))
    status_process.start()
    pool.starmap_async(testing_for_periodicity_worker, args_list)

    pool.close()
    pool.join()
    status_process.join()
    print('[*] Periodicity testing done.')
    for ip in tqdm(source_ips_128):
        if ip not in shared_dict.keys():
            shared_dict[ip] = -1
    print('[*] Extended dictionary with missing IPs (Oneoff, triggered only).')
    print('[*] Mapping IPs with period detection results.')
    df['period_128'] = df.scan_source_128.map(shared_dict)
    print('[*] Operations done - writing back.')
    df.to_parquet(df_file,compression=None)
    print('[*] Done.')
