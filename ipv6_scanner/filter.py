from pathlib import Path

import polars as pl
import pandas as pd
from datetime import datetime

def presplit_filter(df: pl.LazyFrame, column='Timestamp')-> pl.LazyFrame:
    return df.filter(pl.col(column)<=datetime(2023,11,21))

def postsplit_filter(df: pl.LazyFrame)-> pl.LazyFrame:
    return df.filter(pl.col('Timestamp')>datetime(2023,11,21))

def unique_and_resample(df: pl.LazyFrame, subset, keep='first',every='1d',index_column='Timestamp',origin='start'):
    tmp = df.unique(subset=subset, keep=keep).sort(index_column).collect()
    return tmp.group_by_dynamic(index_column=index_column,every=every).agg(pl.count(subset).alias('count')).sort(index_column).with_columns((pl.col('count').cum_sum() / tmp.height).alias(subset))

def concat_frames(dfs: list[pl.LazyFrame], labels: list[str], column: str,columns: list[str]) -> pl.LazyFrame:
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames must match the number of labels.")

    enriched_dfs = [
        df.select(columns).with_columns(pl.lit(label).alias(column))
        for df, label in zip(dfs, labels)
    ]

    return pl.concat(enriched_dfs, how="vertical")

def get_heavy_hitters(dfs: list[pl.LazyFrame], labels: list[str], columns) -> pl.DataFrame:
    heavy_hitter_dfs = {}
    for df,label in zip(dfs,labels):
        tmp_df = df.collect()
        heavy_hitters = tmp_df.select(pl.col('scan_source_128').value_counts(sort=True,normalize=True))\
                        .unnest('scan_source_128')\
                        .filter(pl.col('proportion')>=0.1)\
                        .get_column('scan_source_128')
        heavy_hitter_dfs[label] = tmp_df.filter(pl.col('scan_source_128').is_in(heavy_hitters))

    heavy_hitter_df = pl.DataFrame()
    for telescope in heavy_hitter_dfs.keys():
        tmp = heavy_hitter_dfs[telescope].select(columns).with_columns(pl.lit(f'{telescope.upper()}').alias('Telescope'))
        heavy_hitter_df = pl.concat([heavy_hitter_df,tmp])

    tmp_df = heavy_hitter_df.sort('Timestamp')
    tmp_df = tmp_df.group_by_dynamic('Timestamp',every='1d',group_by=['Telescope','Source_Address']).agg(
        pl.len().alias('Packets [#]'),
        pl.col('Session_ID_128').n_unique().alias('Sessions [#]')
    )

    tmp_df = pl.from_pandas(tmp_df.to_pandas().pivot_table(
        index=['Timestamp', 'Source_Address'],
        columns='Telescope',
        values='Packets [#]',observed=True
        #aggfunc='sum'  # in case there are duplicates
    ).reset_index()).fill_null(0)
    return tmp_df

def process_network_overview(df,interval):
    df = presplit_filter(df)
    return df.group_by_dynamic("Timestamp", every=interval).agg(pl.len().alias("Packets")).collect()

def get_sessions_presplit(df: pl.LazyFrame,resample_interval) -> pl.DataFrame:
    return df.group_by_dynamic('Timestamp',every=resample_interval,start_by='datapoint',group_by='source').agg(
        pl.col('Session_ID_128').n_unique().alias('size')).with_columns(
        pl.col("size").cum_sum().over("source").alias("cumsum"),
        (pl.col("size")/pl.col("size").sum()).over("source").alias("pdf")
    ).with_columns(
        (pl.col("pdf").cum_sum()).over("source").alias("cdf")
    ).collect().upsample('Timestamp',every=resample_interval,group_by='source').with_columns(
            pl.col('source').fill_null(strategy='forward'),
            pl.col('cumsum').fill_null(strategy='forward'),
            pl.col('cdf').fill_null(strategy='forward')
        )

def process_subnet_coverage_plot_data(df, columns):
    df4 = df.select(columns).collect().to_pandas()

    oneoff_df = df4[df4.is_oneoff_128][columns]

    intermittent_df = df4[(~df4.is_oneoff_128) & (df4.period_128 < 1)]
    intermittent_df = intermittent_df[columns]

    periodic_df = df4[(~df4.is_oneoff_128) & (df4.period_128 > 0)]
    periodic_df = periodic_df[columns]

    oneoff_df = df4[df4.is_oneoff_128]
    intermittent_df = df4[(~df4.is_oneoff_128) & (df4.period_128 < 1)]
    periodic_df = df4[(~df4.is_oneoff_128) & (df4.period_128 > 0)]

    possible_subnets_df = pd.DataFrame({'Possible_Subnets': [f'2a05e747{i:04x}' for i in range(2**16)]})

    count_dst = oneoff_df.groupby(oneoff_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_df = pd.merge(possible_subnets_df, count_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    max_count_dst = count_dst['count_dst'].max()
    min_count_dst = count_dst['count_dst'].min()
    mean_count_dst = count_dst['count_dst'].sum() / len(merged_df)
    mean_special_count_dst = count_dst['count_dst'].sum() / len(count_dst)

    max_entry = count_dst[count_dst['count_dst'] == max_count_dst]['fullhex_destination_address'].iloc[0]
    min_entry = count_dst[count_dst['count_dst'] == min_count_dst]['fullhex_destination_address'].iloc[0]

    possible_subnets_df = pd.DataFrame({'Possible_Subnets': [f'2a05e747{i:04x}' for i in range(2**16)]})

    count_dst = intermittent_df.groupby(intermittent_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_df = pd.merge(possible_subnets_df, count_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    max_count_dst = count_dst['count_dst'].max()
    min_count_dst = count_dst['count_dst'].min()
    mean_count_dst = count_dst['count_dst'].sum() / len(merged_df)
    mean_special_count_dst = count_dst['count_dst'].sum() / len(count_dst)

    max_entry = count_dst[count_dst['count_dst'] == max_count_dst]['fullhex_destination_address'].iloc[0]
    min_entry = count_dst[count_dst['count_dst'] == min_count_dst]['fullhex_destination_address'].iloc[0]

    possible_subnets_df = pd.DataFrame({'Possible_Subnets': [f'2a05e747{i:04x}' for i in range(2**16)]})

    count_dst = periodic_df.groupby(periodic_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_df = pd.merge(possible_subnets_df, count_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    max_count_dst = count_dst['count_dst'].max()
    min_count_dst = count_dst['count_dst'].min()
    mean_count_dst = count_dst['count_dst'].sum() / len(merged_df)
    mean_special_count_dst = count_dst['count_dst'].sum() / len(count_dst)

    max_entry = count_dst[count_dst['count_dst'] == max_count_dst]['fullhex_destination_address'].iloc[0]
    min_entry = count_dst[count_dst['count_dst'] == min_count_dst]['fullhex_destination_address'].iloc[0]

    possible_subnets_df = pd.DataFrame({'Possible_Subnets': [f'2a05e747{i:04x}' for i in range(2**16)]})

    count_periodic_df_dst = periodic_df.groupby(periodic_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_periodic_df = pd.merge(possible_subnets_df, count_periodic_df_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    possible_subnets_df = pd.DataFrame({'Possible_Subnets': [f'2a05e747{i:04x}' for i in range(2**16)]})

    count_intermittent_df_dst = intermittent_df.groupby(intermittent_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_intermittent_df = pd.merge(possible_subnets_df, count_intermittent_df_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    count_oneoff_dst = oneoff_df.groupby(oneoff_df['fullhex_destination_address'].str[:12]).size().reset_index(name='count_dst')

    merged_oneoff_df = pd.merge(possible_subnets_df, count_oneoff_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    merged_oneoff_df = pd.merge(possible_subnets_df, count_oneoff_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)
    merged_periodic_df = pd.merge(possible_subnets_df, count_periodic_df_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)
    merged_intermittent_df = pd.merge(possible_subnets_df, count_intermittent_df_dst, left_on='Possible_Subnets', right_on='fullhex_destination_address', how='left').fillna(0)

    sorted_oneoff_df = merged_oneoff_df.sort_values(by='count_dst', ascending=False)
    sorted_periodic_df = merged_periodic_df.sort_values(by='count_dst', ascending=False)
    sorted_intermittent_df = merged_intermittent_df.sort_values(by='count_dst', ascending=False)

    sorted_oneoff_df = sorted_oneoff_df.reset_index(drop=True)
    sorted_periodic_df = sorted_periodic_df.reset_index(drop=True)
    sorted_intermittent_df = sorted_intermittent_df.reset_index(drop=True)

    return (sorted_oneoff_df,sorted_periodic_df,sorted_intermittent_df)

