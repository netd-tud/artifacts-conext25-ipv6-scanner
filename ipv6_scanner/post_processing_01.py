#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from datetime import datetime as dt
import sys
#from parallel_pandas import ParallelPandas
#ParallelPandas.initialize(n_cpu=32, split_factor=1, disable_pr_bar=False)

import warnings
warnings.filterwarnings("ignore")

from ipv6_scanner.config import *

suffix = sys.argv[1]

announcement_file = ANNOUNCEMENT_LOG_FILE
announcement_df = pd.read_csv(announcement_file)

print('[*] Read announcement data.')

df = pd.read_csv(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.csv.gz',sep='|',parse_dates=['Timestamp','Date'])
print('[*] Read dataframe.')

df['Announcement_Timestamp'] = df.Announcement_Period.map(announcement_df.set_index('Id').drop_duplicates('Timestamp_From')['Timestamp_From'])
df['Announcement_Timestamp'] = df.Announcement_Timestamp.astype('datetime64[ns]')

df['announcement_reaction_time'] = df.Timestamp-df.Announcement_Timestamp


print('[*] Determined scan types level 1 dataframe.')

df.sort_values(by=['scan_source_128', 'Timestamp'],inplace=True)

# Calculate interarrival time for each source address
df['Interarrival_Time_128'] = df.groupby('scan_source_128')['Timestamp'].diff()

# Create a function to generate session IDs based on interarrival time
def generate_session_id(interarrival_time):
    global session_id
    if pd.isnull(interarrival_time) or interarrival_time >= pd.Timedelta('3600 seconds'):
        session_id += 1
    return session_id

def unique_value(x):
    u = x.unique()
    if len(u) > 1:
        raise ValueError('Non unique value per group')
    else:
        return u[0]
        
# Initialize session ID
session_id = 1

# Generate session IDs
df['Session_ID_128'] = df.apply(lambda x: generate_session_id(x['Interarrival_Time_128']), axis=1)

print('[*] Session IDs determined for /128.')

df.sort_values(by=['scan_source_64', 'Timestamp'],inplace=True)
# Calculate interarrival time for each source address
df['Interarrival_Time_64'] = df.groupby('scan_source_64')['Timestamp'].diff()
# Initialize session ID
session_id = 1
# Generate session IDs
df['Session_ID_64'] = df.apply(lambda x: generate_session_id(x['Interarrival_Time_64']), axis=1)

print('[*] Session IDs determined for /64.')

df.sort_values(by=['scan_source_48', 'Timestamp'],inplace=True)
# Calculate interarrival time for each source address
df['Interarrival_Time_48'] = df.groupby('scan_source_48')['Timestamp'].diff()
# Initialize session ID
session_id = 1
# Generate session IDs
df['Session_ID_48'] = df.apply(lambda x: generate_session_id(x['Interarrival_Time_48']), axis=1)

print('[*] Session IDs determined for /48.')

oneoff_list = df.groupby(['scan_source_128'])['Session_ID_128'].nunique()[lambda count: count==1].index
df.loc[:,'is_oneoff_128'] = df.scan_source_128.apply(lambda ip: ip in oneoff_list)

oneoff_list = df.groupby(['scan_source_64'])['Session_ID_64'].nunique()[lambda count: count==1].index
df.loc[:,'is_oneoff_64'] = df.scan_source_64.apply(lambda ip: ip in oneoff_list)

oneoff_list = df.groupby(['scan_source_48'])['Session_ID_48'].nunique()[lambda count: count==1].index
df.loc[:,'is_oneoff_48'] = df.scan_source_48.apply(lambda ip: ip in oneoff_list)

print('[*] Determined oneoff scanners.')

potentially_triggered_ips = df[df.announcement_reaction_time<='1h'].scan_source_128.unique()
potentially_triggered = df[df.scan_source_128.isin(potentially_triggered_ips)]#.groupby('scan_source_128').agg(numperiods=('Announcement_Period','nunique'))
potentially_triggered['max_period_count'] = potentially_triggered.scan_source_128.map(df.groupby('scan_source_128')['Announcement_Period'].nunique())
tmp = potentially_triggered[potentially_triggered.announcement_reaction_time<='1h'].groupby('scan_source_128').agg(period_count=('Announcement_Period','nunique'),max_period_count=('max_period_count',lambda x: unique_value(x)))
triggered_ips = tmp[(tmp.period_count>1) & (tmp.period_count>=tmp.max_period_count-1)].index
df['is_triggered_128'] = df.scan_source_128.apply(lambda ip: ip in triggered_ips)

potentially_triggered_ips = df[df.announcement_reaction_time<='1h'].scan_source_64.unique()
potentially_triggered = df[df.scan_source_64.isin(potentially_triggered_ips)]#.groupby('scan_source_128').agg(numperiods=('Announcement_Period','nunique'))
potentially_triggered['max_period_count'] = potentially_triggered.scan_source_64.map(df.groupby('scan_source_64')['Announcement_Period'].nunique())
tmp = potentially_triggered[potentially_triggered.announcement_reaction_time<='1h'].groupby('scan_source_64').agg(period_count=('Announcement_Period','nunique'),max_period_count=('max_period_count',lambda x: unique_value(x)))
triggered_ips = tmp[(tmp.period_count>1) & (tmp.period_count>=tmp.max_period_count-1)].index
df['is_triggered_64'] = df.scan_source_64.apply(lambda ip: ip in triggered_ips)

potentially_triggered_ips = df[df.announcement_reaction_time<='1h'].scan_source_48.unique()
potentially_triggered = df[df.scan_source_48.isin(potentially_triggered_ips)]#.groupby('scan_source_48').agg(numperiods=('Announcement_Period','nunique'))
potentially_triggered['max_period_count'] = potentially_triggered.scan_source_48.map(df.groupby('scan_source_48')['Announcement_Period'].nunique())
tmp = potentially_triggered[potentially_triggered.announcement_reaction_time<='1h'].groupby('scan_source_48').agg(period_count=('Announcement_Period','nunique'),max_period_count=('max_period_count',lambda x: unique_value(x)))
triggered_ips = tmp[(tmp.period_count>1) & (tmp.period_count>=tmp.max_period_count-1)].index
df['is_triggered_48'] = df.scan_source_48.apply(lambda ip: ip in triggered_ips)

print('[*] Determined oneoff scanners.')

addr_map01 = pd.read_csv(ADDR_TYPE_MAP01,sep='|',index_col='ip_addr')
addr_map02 = pd.read_csv(ADDR_TYPE_MAP02,sep='|',names=['ip_addr','addr_type'],index_col='ip_addr')
addr_map = pd.concat([addr_map01,addr_map02])
addr_map.loc[addr_map.index.str.endswith('::'),'addr_type'] = 'full_zero_addr'
df['dest_addr_type'] = df.Destination_Address.map(addr_map.addr_type)
#print(f'DF before filtering Spoki: {len(df)}')
#df = df[((df.TCP_Flags=='nan') | (df.TCP_Flags=='··········S·')) & (~df.Source_Address.str.startswith('2001:67c:254:'))]
print(f'Length of DF: {len(df)}')

df.sort_values('Timestamp',inplace=True)
df = df.dropna(subset=['Hour'])
print('[*] Operations done.')

print('[*] Setting dtypes...')
df['Session_ID_128'] = df['Session_ID_128'].astype('uint32')
df['Session_ID_64'] = df['Session_ID_64'].astype('uint32')
df['Session_ID_48'] = df['Session_ID_48'].astype('uint32')
df['Hour'] = df.Hour.astype('uint8')
df['Minute'] = df.Hour.astype('uint8')
df['Announcement_Period'] = df.Announcement_Period.astype('uint8')
df['AS-Number'] = df['AS-Number'].astype(pd.Int32Dtype())
df['Payload_ByteLength'] = df['Payload_ByteLength'].astype(pd.Int32Dtype())
df['Payload_Length'] = df['Payload_Length'].astype(pd.Int32Dtype())
df['TCP_seq'] = df['TCP_seq'].astype(pd.UInt64Dtype())
df['UDP_src_port'] = df['UDP_src_port'].astype(pd.UInt16Dtype())
df['UDP_dst_port'] = df['UDP_dst_port'].astype(pd.UInt16Dtype())
df['TCP_src_port'] = df['TCP_src_port'].astype(pd.UInt16Dtype())
df['TCP_dst_port'] = df['TCP_dst_port'].astype(pd.UInt16Dtype())

df['is_oneoff_128'] = df['is_oneoff_128'].astype('bool')
df['is_oneoff_64'] = df['is_oneoff_64'].astype('bool')
df['is_oneoff_48'] = df['is_oneoff_48'].astype('bool')

df['is_triggered_128'] = df['is_triggered_128'].astype('bool')
df['is_triggered_64'] = df['is_triggered_64'].astype('bool')
df['is_triggered_48'] = df['is_triggered_48'].astype('bool')

df['DNS_resp_flag'] = df['DNS_resp_flag'].astype(pd.UInt8Dtype())

df['Hop_Limit'] = df['Hop_Limit'].astype('uint8')
df['Next_Header'] = df['Next_Header'].astype('uint8')
df['IP_Version'] = df['IP_Version'].astype('uint8')
df['Frame_Length'] = df['Frame_Length'].astype('uint16')

#print(df.head())
object_columns = df.select_dtypes(include="object").columns
with tqdm(total=len(object_columns)) as pbar:
    for column in object_columns:
        if "Interarrival_Time" in column or "announcement_reaction" in column:
            df[column] = df[column].astype('timedelta64[ns]')
        elif "Timestamp" in column:
            df[column] = df[column].astype('datetime64[ns]')
        else:
            df[column] = df[column].astype(str)
        pbar.update(1)
print('[*] Done.')
print('[*] Writing back parquet file...')

print('[*] Converting categories...')
object_columns = ['Source_Address', 'Protocol',
       'MostSpecificPrefix','scan_source_128', 'scan_source_32',
       'scan_source_48', 'scan_source_64', 'Geo', 'Org','prefix_target',
        'scantool','DNS_id','ICMPv6_Type','QUIC_version','TCP_ack','Flow_Label','TCP_Flags']
with tqdm(total=len(object_columns)) as pbar:
    for column in object_columns:
        df[column] = df[column].astype('category')
        pbar.update(1)

print('[*] Done.')

print(f'[*] Writing back parquet file with categories:')
print(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.parquet')
df.to_parquet(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.parquet',compression=None)

#df.to_csv(f'{working_dir}/bcix-telescope-data-complete.csv.gz',index=False,sep='|',compression='gzip')
print('[*] Done.')
