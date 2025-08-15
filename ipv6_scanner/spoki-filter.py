import polars as pl

from ipv6_scanner.config import *
import sys

if __name__=='__main__':
    suffix = sys.argv[1]
    df = pl.read_parquet(f'{PROCESSED_DATA_DIR}/telescope-t{suffix}_data.parquet')
    print(df.head())
    print(len(df))
    df = df.filter(
        ((pl.col("TCP_Flags").is_null()) | (pl.col("TCP_Flags") == "··········S·"))
        & (~pl.col("Source_Address").cast(pl.Utf8).str.starts_with("2001:67c:254:"))
        & (~pl.col("Source_Address").cast(pl.Utf8).str.starts_with('2a01:6420:8031'))
    )
    print(len(df))