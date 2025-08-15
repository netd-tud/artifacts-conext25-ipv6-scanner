import polars as pl

from ipv6_scanner.config import *
import sys

if __name__=='__main__':
    suffix = sys.argv[1]
    print(f'[*] Loading Dataframe T{suffix}')
    df = pl.read_parquet(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.parquet')
    #print(df.head())
    print(f'DF size:{len(df)}')
    #print(df.select(['Source_Address','TCP_Flags']))
    print(df['TCP_Flags'].value_counts())
    print(f'[*] Filtering data...')
    df = df.filter(
        ((pl.col("TCP_Flags")=='nan') | (pl.col("TCP_Flags") == "··········S·"))
        & (~pl.col("Source_Address").cast(pl.Utf8).str.starts_with("2001:67c:254:"))
        & (~pl.col("Source_Address").cast(pl.Utf8).str.starts_with('2a01:6420:8031'))
    )
    print(f'DF size:{len(df)}')
    print(f'[*] Writing back...')
    df.write_parquet(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.parquet')
    print(f'[*] Done.')
    #print(len(df))