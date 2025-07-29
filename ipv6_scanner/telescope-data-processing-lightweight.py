#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime as dt

from time import sleep

import gzip

import os
import maxminddb
from functools import lru_cache

import geoip2.database
from pathlib import Path

from datetime import datetime, timedelta

from textwrap import wrap

import subprocess
import pyasn
import ipaddress

import multiprocessing as mp

#from parallel_pandas import ParallelPandas
#ParallelPandas.initialize(n_cpu=8, split_factor=1, disable_pr_bar=False)

import warnings
warnings.filterwarnings("ignore")
# In[2]:




def assign_tool(payload):
    # Payload starts with these bytes...
    id_scanner_mapping_startbytes = {
        '79727036' : 'Yarrp6',
        '06536361' : '6Scan',
        '4c4f5645' : 'Love',
        '68747470733a2f2f365365656b732e6769746875622e696f2f' : '6Seek',
        '68747470733a2f2f6c61622d616e742e6769746875622e696f2f365365656b732f' : '6Seek'
    }
    # ...or ends with these
    id_scanner_mapping_endbytes = {
        '48526f75746536' : 'HRoute6',
        '747261636536' : 'Trace6'
    }
    # check for complete payload (6Seek)
    if payload in id_scanner_mapping_startbytes.keys():
        return id_scanner_mapping_startbytes[payload]
    # check for first 4 byte ID
    if payload[:8] in id_scanner_mapping_startbytes.keys():
        return id_scanner_mapping_startbytes[payload[:8]]
    # check if the last bytes end with specific ID
    if payload[-14:] in id_scanner_mapping_endbytes.keys():
        return id_scanner_mapping_endbytes[payload[-14:]]
    if payload[-12:] in id_scanner_mapping_endbytes.keys():
        return id_scanner_mapping_endbytes[payload[-12:]]
    # ascii characters in ascending order are produced by standard traceroute tools
    if '61626364656667' in payload or '40414243444546474849' in payload:
        return 'Traceroute'
    # part of the ripe atlas payload padding string that can be found in most measurements
    # full string would be "http://atlas.ripe.net Atlas says Hi!"
    if '687474703a2f2f61' in payload:
        return 'RIPEAtlasProbe'
    else:
        return 'Other'

# In[3]:
def most_specific_prefix(address, prefixes):
    """
    Find the most specific prefix for a given IPv6 address among a list of prefixes.

    Args:
    - address: IPv6 address (string)
    - prefixes: List of IPv6 prefixes (strings)

    Returns:
    - The most specific prefix (string)
    """
    if ',' in address:
        addresses = address.split(',')
        for addrs in addresses:
            if '2a05:e747' in addrs:
                address = addrs
                break
    # Convert the address to an IPv6Address object
    addr = ipaddress.IPv6Address(address)

    # Convert each prefix to an IPv6Network object and find the most specific one
    most_specific_prefix = None
    max_length = -1
    for prefix_str in prefixes:
        prefix = ipaddress.IPv6Network(prefix_str)
        if addr in prefix and prefix.prefixlen > max_length:
            most_specific_prefix = prefix
            max_length = prefix.prefixlen

    return str(most_specific_prefix)

def address_classification(address):
    return str(subprocess.check_output(['./process-flows/ipv6toolkit/addr6', "-a",address])).split('=')[-2]
    
# get bytes either before the encoded ip address (mode==0) or after (mode==1)
def splitIPAdrFromOtherBytes(payload,mode,startbytes):
    # length (in hexbytes) of the ipv6 addr encoded in the payload (16bytes = 32hexbytes = 128bit)
    length = 32
    startbyteslength = len(startbytes)
    # number of remaining hex characters after splitting
    taillength = length-startbyteslength
    # get bytes before ipadr
    if mode==0:
        return payload.split(startbytes)[0]
    else:
        try:
            return payload.split(startbytes)[1][taillength:]
        except:
            return ''

def expand_ipv6(ipv6_addr):
    # split the IPv6 address into its individual segments
    segments = ipv6_addr.split(':')

    # check if the address contains double colons
    if '' in segments:
        # find the index where the double colons appear
        index = segments.index('')
        # calculate the number of segments that need to be added
        num_segments = 8 - len(segments) + 1
        # insert the necessary number of zero segments at the index
        segments[index:index+1] = ['0000'] * num_segments
    # expand each segment to 4 digits
    for i in range(len(segments)):
        segment = segments[i]
        if len(segment) < 4:
            segments[i] = segment.rjust(4, '0')

    # join the expanded segments back together into a full IPv6 address
    return ':'.join(segments)
def full_hex(ipv6_addr):
    expanded = expand_ipv6(ipv6_addr)
    res = ''
    for char in expanded:
        if char==':':
            continue
        res+=char
    return res
# In[4]:


def hex2text(hexstring,encoding):
    try:
        return bytes.fromhex(hexstring).decode(encoding)
    except:
        return '?'

"""
Get country geolocation from Maxmind.
"""


class Maxmind:
    def __init__(self, databasefolder):
        dbpath = Path(databasefolder)
        if not dbpath.is_dir():
            print(f"Not a directory: '{databasefolder}'")
            exit()

        self.databasefolder = databasefolder

        self.current_db = None
        self.reader = None

        self.last_checked = datetime.now()

        self.check_interval = timedelta(days=1)

    # -- load database -------------------------------------------------------

    def initialize(self):
        db = self.newest_db()
        if db is None:
            return False
        else:
            self.load_db(db)
            if self.reader is None:
                return False
        return True

    def try_reload(self):
        db = self.newest_db()
        if db is not None and db != self.current_db:
            self.load_db(db)
            self.last_checked = datetime.now()

    def should_check_again(self):
        now = datetime.now()
        if self.last_checked - now >= self.check_interval:
            return True
        else:
            return False

    def newest_db(self):
        pattern = os.path.join(self.databasefolder, "*.mmdb")
        database_list = glob.glob(pattern)
        if len(database_list) == 0:
            return None
        newest_database = max(database_list, key=os.path.getctime)
        self.last_checked = datetime.now()
        return newest_database

    def load_db(self, database):
        # print(f"loading {database}")
        try:
            self.reader = geoip2.database.Reader(database)
            self.current_db = database
        except FileNotFoundError as fnfe:
            print(f"Not a valid file: '{database}' ({fnfe})")
        except PermissionError as pr:
            print(f"Permission error: '{database}' ({pr})")
        except maxminddb.errors.InvalidDatabaseError as ide:
            print(f"Invalid database: '{database}' ({ide})")

    # -- query ---------------------------------------------------------------

    def query(self, ip):
        assert self.reader is not None, "ERR: Load a database before querying."
        try:
            res = self.reader.country(ip)

            name = res.country.name
            iso = res.country.iso_code
            # Not in this database:
            # lat = res.location.latitude
            # lon = res.location.latitude

            return (iso, name)

        except ValueError:
            return (None, None)
        except geoip2.errors.AddressNotFoundError:
            return (None, None)
        except TypeError:
            return (None, None)


# In[12]:


"""
Get AS and prefix info.
"""

class IPasn:
    def __init__(self, databasefolder):
        dbpath = Path(databasefolder)
        if not dbpath.is_dir():
            print(f"Not a directory: '{databasefolder}'")
            exit()

        self.databasefolder = databasefolder

        self.current_db = None
        self.reader = None

        self.last_checked = datetime.now()

        self.check_interval = timedelta(days=1)

    # -- load database -------------------------------------------------------

    def initialize(self):
        db = self.newest_db()
        if db is None:
            return False
        else:
            self.load_db(db)
            if self.reader is None:
                return False
        return True

    def try_reload(self):
        db = self.newest_db()
        if db is not None and db != self.current_db:
            self.load_db(db)
            self.last_checked = datetime.now()

    def should_check_again(self):
        now = datetime.now()
        if self.last_checked - now >= self.check_interval:
            return True
        else:
            return False

    def newest_db(self):
        pattern = os.path.join(self.databasefolder, "ipasn_*.gz")
        database_list = glob.glob(pattern)
        if len(database_list) == 0:
            return None
        newest_database = max(database_list, key=os.path.getctime)
        self.last_checked = datetime.now()
        return newest_database

    def load_db(self, database):
        try:
            reader = pyasn.pyasn(database)
            self.reader = reader
            self.current_db = database

        except FileNotFoundError as fnfe:
            print(f"Not a valid file: '{database}' ({fnfe})")
        except PermissionError as pr:
            print(f"Permission error: '{database}' ({pr})")

    # -- query ---------------------------------------------------------------

    def query(self, ip):
        assert self.reader is not None, "ERR: Load a database before querying."
        try:
            asn, prefix = self.reader.lookup(ip)

            if asn is not None:
                asn = int(asn)

            return (asn, prefix)

        except ValueError as ve:
            #print(f"No data for address: '{ip}' ({ve})")
            return (None, None)
        except TypeError as te:
            #print(f"Not a valid type: '{ip}' ({te})")
            return (None, None)


# In[13]:


"""
Get AS organization names.
"""

class ASname:
    def __init__(self, databasefolder):
        dbpath = Path(databasefolder)
        if not dbpath.is_dir():
            print(f"Not a directory: '{databasefolder}'")
            exit()

        self.databasefolder = databasefolder

        self.current_db = None
        self.reader = None

        self.last_checked = datetime.now()

        self.check_interval = timedelta(days=1)

    # -- load database -------------------------------------------------------

    def initialize(self):
        db = self.newest_db()
        if db is None:
            return False
        else:
            self.load_db(db)
            if self.reader is None:
                return False
        return True

    def try_reload(self):
        db = self.newest_db()
        if db is not None and db != self.current_db:
            self.load_db(db)
            self.last_checked = datetime.now()

    def should_check_again(self):
        now = datetime.now()
        if self.last_checked - now >= self.check_interval:
            return True
        else:
            return False

    def newest_db(self):
        pattern = os.path.join(self.databasefolder, "asnames-*.csv.gz")
        database_list = glob.glob(pattern)
        if len(database_list) == 0:
            return None
        newest_database = max(database_list, key=os.path.getctime)
        self.last_checked = datetime.now()
        return newest_database

    def load_db(self, database):
        try:
            self.reader = (
                pd.read_csv(
                    database,
                    sep="|",
                    header=None,
                    index_col=[0],
                    usecols=[0, 1],
                )
                .squeeze("columns")
                .to_dict()
            )
            self.current_db = database
        except FileNotFoundError as fnfe:
            print(f"Not a valid file: '{database}' ({fnfe})")
        except PermissionError as pr:
            print(f"Permission error: '{database}' ({pr})")

    # -- query ---------------------------------------------------------------

    def query(self, ip):
        assert self.reader is not None, "ERR: Load a database before querying."
        try:
            if ip in self.reader:
                return self.reader[ip]
            else:
                return None

        except ValueError as ve:
            #print(f"No data for address: '{ip}' ({ve})")
            return None
        except TypeError as te:
            #print(f"Not a valid type: '{ip}' ({te})")
            return (None, None)


# In[14]:



maxmindpath = './data/external/maxmind/'
ipasnpath = './data/external/ipasn/'
asnamepath = './data/external/asname/'
fields = 'Source_Address'

@lru_cache(maxsize=10000)
def lookup_asn(asndb, addr):
    # asn, prefix = ipasn.query(saddr)
    asn, _ = asndb.query(addr)
    return asn

@lru_cache(maxsize=10000)
def lookup_prefix(asndb, addr):
    # asn, prefix = ipasn.query(saddr)
    _, prefix = asndb.query(addr)
    return prefix

@lru_cache(maxsize=10000)
def lookup_geo(geodb, addr):
    # iso, _ = mm.query(saddr)
    iso, _ = geodb.query(addr)
    return iso

@lru_cache(maxsize=10000)
def lookup_org(orgdb, asn):
    # org = asname.query(asn)
    return orgdb.query(asn)


def get_infos(maxmindpath, ipasnpath, asnamepath, df):

    # -- check paths ---------------------------------------------------------

    if not Path(maxmindpath).is_dir():
        print("ERR: maxmind path is not a directory!")
        return
    
    if not Path(ipasnpath).is_dir():
        print("ERR: please select a directory with --ipasn-db")
        return

    if not Path(asnamepath).is_dir():
        print("ERR: please select a directory with --asname-db")
        return
    

# # -- initialize meta data stuff ------------------------------------------

    # Initialize Maxmind database
    mm = Maxmind(maxmindpath)
    if not mm.initialize():
        print(f"ERR: could not load Maxmind database from {maxmindpath}")
        return
    
    # Initialize pyasn database.
    ipasn = IPasn(ipasnpath)
    if not ipasn.initialize():
        print(f"ERR: could not load ipasn database from {ipasnpath}")
        return

    # Initialize AS name database.
    asname = ASname(asnamepath)
    if not asname.initialize():
        print(f"ERR: could not load as-to-name database from {asnamepath}")
        return

    df['Geo'] = df['Source_Address'].apply(lambda ip: lookup_geo(mm, ip))
    df['AS-Number'] = df['Source_Address'].apply(lambda ip: lookup_asn(ipasn, ip))
    df['prefix_target'] = df['Source_Address'].apply(lambda ip: lookup_prefix(ipasn, ip))
    df['Org'] = df['AS-Number'].apply(lambda asn: lookup_org(asname, asn))
    total = len(df.index)
    pre_missing = df['prefix_target'].isnull().sum()
    cou_missing = df['Geo'].isnull().sum()
    asn_missing = df['AS-Number'].isnull().sum()
    org_missing = df['Org'].isnull().sum()
    #print(f"total          = {total:>7}")
    #print(
    #    f"missing country = {cou_missing:>7} ({round(cou_missing / total * 100, 2)}%)"
    #)

def load_join_and_save(files,columns,prefixes,announcement_df,save_dir):
    df = pd.DataFrame(columns=columns)
    #with tqdm(total=len(files)) as pbar:
    basefile = files[0].split('/')[-1]
    for file in files:
        try:
            tmp = pd.read_csv(file,sep='|', names=columns,on_bad_lines='warn')
        except (pd.errors.EmptyDataError, EOFError):
        #except:
            #pbar.update(1)
            continue
        df = pd.concat([df, tmp], ignore_index=True)
        #pbar.update(1)
    df = df[df.Source_Address!='2a01:6420:8030:1::82']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df.Timestamp.dt.date
    df['Hour'] = df.Timestamp.dt.hour
    df['Minute'] = df.Timestamp.dt.minute
    df['Payload'] = df.Payload.astype(str)
    df['MostSpecificPrefix'] = df.Destination_Address.apply(most_specific_prefix,args=(prefixes,))
    df['scantool'] = df.Payload.apply(assign_tool)
    #df['address_type'] = df.Destination_Address.p_apply(address_classification)
    df['scan_source_128'] = df.Source_Address.apply(full_hex)
    df['fullhex_destination_address'] = df.Destination_Address.apply(full_hex)
    df['scan_source_32'] = df.scan_source_128.apply(lambda ip: ip[:8])
    df['scan_source_48'] = df.scan_source_128.apply(lambda ip: ip[:12])
    df['scan_source_64'] = df.scan_source_128.apply(lambda ip: ip[:16])
    #df['dest_addr_segments_length_1'] = df.fullhex_destination_address.apply(lambda addr: wrap(addr,1))
    #df['dest_addr_segments_length_2'] = df.fullhex_destination_address.apply(lambda addr: wrap(addr,2))
    #df['dest_addr_segments_length_4'] = df.fullhex_destination_address.apply(lambda addr: wrap(addr,4))
    df.sort_values(by='Timestamp',inplace=True)
    df = pd.merge_asof(df, announcement_df[['Id','Timestamp_From']].sort_values('Timestamp_From'), 
                          left_on='Timestamp', right_on='Timestamp_From').drop('Timestamp_From',axis=1)
    df.rename(columns={'Id':'Announcement_Period'},inplace=True)
    get_infos(maxmindpath, ipasnpath, asnamepath,df)

    df.to_csv(f'{save_dir}/{basefile}',sep='|',index=False)

# In[16]:
suffix=sys.argv[1]

working_dir = '/home/koch/process-flows'
#files = glob.glob('./flowdaten-archive/bcix*')
files = glob.glob(f'{working_dir}/csv_6{suffix}/bcix*')

files.sort()

# In[15]:


columns=['Timestamp', 'Frame_Length', 'IP_Version', 'Flow_Label', 'Payload_Length', 
                           'Next_Header', 'Hop_Limit', 'Source_Address', 'Destination_Address', 'Protocol', 
                           'UDP_src_port', 'UDP_dst_port', 'TCP_src_port', 'TCP_dst_port', 'TCP_payload', 'TCP_seq', 'TCP_ack', 'TCP_Flags', 
                           'QUIC_version', 'ICMPv6_Type','Payload','Payload_Text','Payload_ByteLength',
                          'DNS_id','DNS_qry_name','DNS_resp_flag']

announcement_file = f'{working_dir}/prefix-announcements.csv'
announcement_df = pd.read_csv(announcement_file)
announcement_df['Timestamp_From'] = pd.to_datetime(announcement_df['Timestamp_From'])
announcement_df['Timestamp_To'] = pd.to_datetime(announcement_df['Timestamp_To'])
prefixes = announcement_df.Prefix.unique()

def status_bar(directory,total):
    with tqdm(total=total) as pbar:
        current_status = len(glob.glob(f'{directory}/bcix*'))
        pbar.update(current_status)
        old=current_status
        while total>current_status:
            sleep(5)
            current_status = len(glob.glob(f'{directory}/bcix*'))
            pbar.update(current_status-old)
            old=current_status
            
def parallel_processing(files, columns, prefixes, announcement_df, output_dir, chunk_size=10,n_cpus=8):
    # Split the list of files into chunks
    file_chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]
    
    # Create a multiprocessing pool
    pool = mp.Pool(processes=n_cpus)
    
    # Total number of chunks
    total_chunks = len(file_chunks)

    # Start status bar process
    status_process = mp.Process(target=status_bar, args=(output_dir, total_chunks))
    status_process.start()
    
    pool.starmap_async(load_join_and_save, [(chunk, columns, prefixes, announcement_df, output_dir) for chunk in file_chunks])
    # Close the pool to release resources
    pool.close()
    pool.join()
    
    status_process.join()

def append_df_to_gzip(filename, data):
    # Convert DataFrame to CSV string without headers
    csv_data = data.to_csv(index=False, header=False,sep='|', lineterminator='\n')
    
    # Append CSV data to gzip file
    with gzip.open(filename, 'ab') as f_out:
        f_out.write(csv_data.encode())

# Usage
parallel_processing(files, columns, prefixes, announcement_df, f'./data/raw/t{suffix}_raw')

files_to_join = glob.glob(f'./data/raw/t{suffix}_raw/*')

files_to_join.sort()

df_file = f'./data/processed/telescope-t{suffix}_data.csv.gz'

new_df = None
for i in tqdm(range(len(files_to_join))):
    if i==0:
        new_df = pd.read_csv(files_to_join[i],sep='|')
    else:
        tmp = pd.read_csv(files_to_join[i],sep='|')
        new_df = pd.concat([new_df,tmp],ignore_index=True)
new_df.to_csv(df_file,sep='|',index=False,compression='gzip',quotechar='"')
    

