from pathlib import Path

import typer
import os
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import matplotlib as mpl

import seaborn as sns
from matplotlib.patches import Patch,Rectangle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.markers import MarkerStyle

from ipv6_scanner.filter import *
from ipv6_scanner.config import *
from ipv6_scanner.helper import *

app = typer.Typer()

@timeit
def new_prefixes(df, fig_name, fig_size,columns,**kwargs):
    pickle_path = os.path.join(INTERIM_DATA_DIR, f"{fig_name}_cached.pkl")

    def process_df(df,columns):
        return unique_and_resample(presplit_filter(df.select(columns)), 'prefix_target')

    df = load_or_process_pickle(pickle_path, process_df, df,columns)
    
    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)

    ax.plot(df['Timestamp'],df['prefix_target'])

    ax.set_ylim(0,1)

    ax.set_xlim(pd.to_datetime('2023-08-24'), pd.to_datetime('2023-11-21'))
    ax.set_xlabel('Time [D]')
    ax.set_ylabel('CDF')
    ax.grid(axis='y')

    minorticks= pd.date_range('2023-08-24', '2023-11-21',freq='W')
    ax.set_xticks(minorticks, minor=True)

    plt.setp(ax, xticks=[pd.to_datetime('2023-09-01'),pd.to_datetime('2023-10-01'),pd.to_datetime('2023-11-01')], xticklabels=['Sep','Oct','Nov'])
    plt.xticks(rotation=0,ha='center')

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.axvspan('2023-08-24', '2023-09-07', color='red', alpha=0.3)


    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def aggregated_telescopes_full_period(dfs, labels, fig_name, fig_size, resample_interval, label_column, columns, **kwargs):
    df = concat_frames(dfs, labels, label_column, columns)

    base_name = os.path.join(INTERIM_DATA_DIR, fig_name)

    new_as_df = load_or_process_pickle(f"{base_name}_as.pkl",
                                       unique_and_resample, df, 'AS-Number', every=resample_interval)
    new_sources128_df = load_or_process_pickle(f"{base_name}_src128.pkl",
                                               unique_and_resample, df, 'scan_source_128', every=resample_interval)
    new_sources64_df = load_or_process_pickle(f"{base_name}_src64.pkl",
                                              unique_and_resample, df, 'scan_source_64', every=resample_interval)
    new_sessions128_df = load_or_process_pickle(f"{base_name}_sess128.pkl",
                                                unique_and_resample, df, 'Session_ID_128', every=resample_interval)
    new_sessions64_df = load_or_process_pickle(f"{base_name}_sess64.pkl",
                                               unique_and_resample, df, 'Session_ID_64', every=resample_interval)
    packets_df = load_or_process_pickle(f"{base_name}_packets.pkl",
        lambda: df.sort('Timestamp')
                  .group_by_dynamic(index_column='Timestamp', every=resample_interval)
                  .agg(pl.len().alias("count"))
                  .collect()
                  .with_columns((pl.col("count").cum_sum() / df.collect().height).alias("Packets [#]"))
    )

    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)
    colors = ["#375E97", "#FB6542", "#c1195c", "#37975e","#a742f5","#42f5e6"]
    linestyles = ['--','solid','-.','dotted']

    ax.plot(packets_df['Timestamp'],packets_df['Packets [#]'],label='Packets [#]',linestyle=linestyles[0],color=colors[3])
    ax.plot(new_as_df['Timestamp'],new_as_df['AS-Number'],label='ASes [#]',linestyle=linestyles[1],color=colors[2])
    ax.plot(new_sources128_df['Timestamp'],new_sources128_df['scan_source_128'],label='Src /128 [#]',linestyle=linestyles[1],color=colors[1])
    ax.plot(new_sources64_df['Timestamp'],new_sources64_df['scan_source_64'],label='Src /64 [#]',linestyle=linestyles[0],color=colors[1])
    ax.plot(new_sessions128_df['Timestamp'],new_sessions128_df['Session_ID_128'],label='Sessions /128 [#]',linestyle=linestyles[1],color=colors[0])
    ax.plot(new_sessions64_df['Timestamp'],new_sessions64_df['Session_ID_64'],label='Sessions /64 [#]',linestyle=linestyles[0],color=colors[0])

    ax.set_ylim(0,1)

    ax.set_xlim(pd.to_datetime('2023-08-24'), pd.to_datetime('2024-07-02'))
    ax.set_xlabel('Time [D]')
    ax.set_ylabel('\nCDF')
    ax.grid(axis='y')
    ax.set_yticks([0,0.5,1])

    #ax.set_yticklabels([f'{int(y)}K' if y>0 else '0' for y in ax.get_yticks()/1000])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    plt.xticks(rotation=0,ha='center')

    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    #ax.axvspan('2023-08-24', '2023-09-07', color='red', alpha=0.3)
    ax.axvline(pd.to_datetime('2023-11-22'), color='gray', alpha=0.3, linestyle='dashed')

    handles, labels = ax.get_legend_handles_labels()
    #for handle in handles:
    #    handle.set_linewidth(2.0)
    leg = ax.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5,1.4),columnspacing=1,ncol=3,fontsize=9,handlelength=2.5)

    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)
    
@timeit
def heavy_hitter_bubble(dfs,labels, fig_name, fig_size,columns,vertical_dates,**kwargs):
    def log_scale_sizes(counts, scale=20):
        counts = np.array(counts)
        return scale * np.log1p(counts)  # log1p to avoid log(0)

    pickle_path = os.path.join(INTERIM_DATA_DIR, f"{fig_name}_cached.pkl")

    def process_dfs(dfs,labels,columns):
        return get_heavy_hitters(dfs,labels,columns)

    heavy_hitter_df = load_or_process_pickle(pickle_path, process_dfs, dfs, labels, columns)
    
    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)
    min_date=pd.to_datetime('2023-08-24')
    max_date=pd.to_datetime('2024-07-02')

    colors = plt.get_cmap('tab10_r')
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"
    ]

    colormap = {'2001:253:ff:4301:3673:79ff:fe91:48b8':colors[0],
    '2001:4ca0:108:42:0:1:3:28':colors[1],
    '2001:550:9005:1000::11':colors[2],
    '2001:550:9005:2000::11':colors[3],
    '2001:550:9005:e000::11':colors[4],
    '240b:4002:12:7f00:35f2:ab24:b273:1622':colors[5],
    '240d:c000:2010:1a42:0:98e7:da67:1fb7':colors[6],
    '240e:c2:1800:84:0:1:3:2':colors[7],
    '2605:6400:10:6a8:bc7a:408:1ad:e7e6':colors[8],
    '240b:4001:112:7b00:79e3:688a:710c:6051':colors[9]}

    telescopes = ["T1", "T2", "T3", "T4"]

    for pos, addr in enumerate(heavy_hitter_df['Source_Address'].unique().to_list()):
        for i, scope in enumerate(telescopes):
            sizes = log_scale_sizes(heavy_hitter_df.filter(pl.col('Source_Address')==addr)[scope])
            xs = heavy_hitter_df.filter(pl.col('Source_Address')==addr)['Timestamp'].to_list()
            ys = [i] * len(xs)

            if scope=='T3' and addr == "2001:550:9005:1000::11":
                ax.scatter(
                    y=ys,
                    x=xs,
                    s=sizes,
                    color=colormap[addr],
                    edgecolors='black',
                    linewidths=0.1,
                    alpha=0.6,
                    marker=MarkerStyle("o", fillstyle="left")
                )
            elif scope=='T3' and addr == "2001:550:9005:2000::11":
                ax.scatter(
                    y=ys,
                    x=xs,
                    s=sizes,
                    color=colormap[addr],
                    edgecolors='black',
                    linewidths=0.1,
                    alpha=0.6,
                    marker=MarkerStyle("o", fillstyle="right")
                )
            else:
                ax.scatter(
                    y=ys,
                    x=xs,
                    s=sizes,
                    color=colormap[addr],
                    edgecolors='black',
                    linewidths=0.1,
                    alpha=0.6,
                )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    # ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_ticks([0, 1, 2, 3])
    ax.yaxis.set_ticklabels(["T1", "T2", "T3", "T4"])

    # ax.set_ylim([10**0, 10**6])

    ax.set_xlim([min_date, max_date])
    ax.set_ylim([-0.5, 3.3])

    ax.axvline(vertical_dates[0], color="grey", linestyle="--", linewidth=1)

    ax.set_ylabel("Telescope")
    ax.set_xlabel("Time [D]")

    plt.gca().invert_yaxis()

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def network_traffic_overview(dfs,labels, fig_name, fig_size,interval,**kwargs):
    def process_dfs(df, interval):
        return process_network_overview(df,interval)

    plt.rc("font", size=12)
    fig,axes = fig_ax(fig_size,nrows=2,ncols=2,sharey=True)

    i = 0
    for ax, data, title in zip(fig.axes, dfs, labels):
        pickle_path = os.path.join(INTERIM_DATA_DIR, f"{fig_name}_{title[:2]}.pkl")
        
        df = load_or_process_pickle(pickle_path, process_dfs, data,interval)

        ax.plot(df["Timestamp"], df["Packets"], linestyle='none',
                marker='.', c='black', alpha=0.7, ms=2)

        if i%2==0:
            ax.set_ylabel('Packets [#]')
        i+=1
        ax.set_yscale('log')
        ax.set_ylim(1, 10**6)
        ax.set_title(title, fontsize=12)

        minorticks = pl.date_range(start=pl.datetime(2023, 8, 24), 
                                end=pl.datetime(2023, 11, 21), 
                                interval="1w", 
                                eager=True)
        ax.set_xticks(minorticks.to_list(), minor=True)
        ax.set_xlim(pd.to_datetime('2023-08-24'),pd.to_datetime('2023-11-21'))
    # Set the ticks and ticklabels for all axes
    plt.setp(axes, xticks=[pd.to_datetime('2023-09-01'),pd.to_datetime('2023-10-01'),pd.to_datetime('2023-11-01')], xticklabels=['Sep','Oct','Nov'])

    fig.supxlabel(f'Time [{interval[-1].upper()}]',fontsize=12,y=0.05)

    fig.tight_layout()

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def taxonomy_plot_before_split(df,fig_name,fig_size):
    # Define the order of temporal_behavior categories
    temporal_order = ['periodic', 'intermittent', 'one-off']

    # Define the order for address selection types
    address_order = ['structured', 'random', 'undetermined']

    # Convert temporal_behavior to a categorical type with the specified order
    df['temporal_behavior'] = pd.Categorical(df['temporal_behavior'], categories=temporal_order, ordered=True)

    # Categorical-Datentyp für telescope
    df['telescope'] = pd.Categorical(df['telescope'])

    # Dummy column for groupby to work
    df['Dummy'] = np.ones(len(df))

    # Count combinations
    grouped = df.groupby(by=['temporal_behavior', 'telescope', 'address_selection'],observed=True).size().unstack(fill_value=0)

    # Get unique levels for telescope and address_selection
    temporal_levels = df['temporal_behavior'].cat.categories  # Use categories from categorical data
    telescope_levels = df['telescope'].unique()
    # Use address_order to ensure the correct order of address_types
    address_types = address_order  # Set address types in desired order

    # Create a full DataFrame to account for all possible combinations
    index = pd.MultiIndex.from_product([temporal_levels, telescope_levels], names=['temporal_behavior', 'telescope'])
    columns = address_types
    full_df = pd.DataFrame(index=index, columns=columns).fillna(0)

    # Insert the counted data into the DataFrame
    full_df.update(grouped)

    # Custom colors for the bar chart
    colors = ["#FB6542", "#c1195c", "#375E97"]

    # Create subplots with the adjusted figsize
    sns.set(context="talk")
    nxplots = len(temporal_levels)  # Number of rows in the subplot matrix
    nyplots = len(telescope_levels)  # Number of columns in the subplot matrix
    fig, axes = plt.subplots(nxplots, nyplots, sharey=True, sharex=True, figsize=fig_size)

    plt.rc("font", size=9)

    # Plotting the data
    for a, temporal in enumerate(temporal_levels):
        for i, telescope in enumerate(telescope_levels):
            # Extract data for the current combination
            data = full_df.loc[(temporal, telescope)]
            
            # Plot the bar chart in the order specified in address_order
            bars = axes[a, i].bar(address_types, data[address_types], color=colors[:len(data)])
            
            # Logarithmic y-axis (check if positive values are present)
            if (data > 0).any():
                axes[a, i].set_yscale('log')
                axes[a, i].set_ylim(1, 1000000)

                # Set custom ticks and labels
                ticks = [10, 100, 1000, 10000, 100000]
                axes[a, i].set_yticks(ticks)

                # Set the y-axis tick label font size
                axes[a, i].tick_params(axis='y', labelsize=9)  # Set y-axis tick label font size

            # Remove titles for individual subplots
            axes[a, i].set_title('')
            # Remove x-axis ticks (optional, depending on desired layout)
            axes[a, i].xaxis.set_ticks([])

    # Add telescope labels above the entire subplot grid
    for i, telescope in enumerate(telescope_levels):
        fig.text(
            0.21 + 0.205 * i,  # Adjust the positioning for better centering
            0.88,             # Position just above the plots (between 0 and 1)
            telescope,        # The telescope label
            ha='center',      # Centered horizontally
            va='top',         # Align at the top vertically
            fontsize=9       # Adjust the font size
        )

    # Set axis labels for the array of Axes
    for a, temporal in enumerate(temporal_levels):
        fig.text(
            0.92,                      # X-position (a little to the left of the subplots)
            0.74 - 0.26 * a,           # Y-position for each label (adjusted for each row)
            temporal,                  # The label (temporal behavior)
            ha='center',               # Center horizontally
            va='center',               # Center vertically
            rotation=-90,              # Rotate the text to be vertical
            fontsize=9                # Adjust font size
        )

    # Create space for the legend
    fig.subplots_adjust(top=0.85)

    # Add the legend and place it below the plot
    patches = [Patch(color=colors[i], label=address_order[i]) for i in range(len(address_order))]
    legend = fig.legend(
        handles=patches,
        loc="lower center",               # Set legend to lower center
        bbox_to_anchor=(0.5, 0.04),      # Position the legend below the plot
        ncol=len(address_order),          # Number of columns in the legend
        frameon=False,                    # Remove the legend frame
        prop={'size': 9}                # Set font size for legend text
    )

    # Set marker size to make the patches smaller
    for patch in legend.get_patches():
        patch.set_height(10)  # Adjust the height of legend patches
        patch.set_width(20)   # Adjust the width of legend patches

    fig.text(0.02, 0.48, 'Sessions [#]', va='center', rotation='vertical', fontsize=9)
    fig.text(0.42, 0.04, 'Address selection', va='center', fontsize=9)
    fig.text(0.5, 0.916, 'Network telescopes', ha='center', va='top', fontsize=9)
    fig.text(0.96, 0.48, 'Temporal behavior', ha='center', va='center', rotation=-90, fontsize=9)
    
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)
    sns.reset_defaults()
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

@timeit
def taxonomy_plot_during_split(df,fig_name,fig_size):
    # Categorical-Datentyp für network_selection
    df['network_selection'] = pd.Categorical(df['network_selection'])

    # Define the order for address selection types
    address_order = ['structured', 'random', 'unknown']

    # Dummy-Spalte, damit das groupby funktioniert
    df['Dummy'] = np.ones(len(df))

    # Zählen der Kombinationen
    grouped = df.groupby(by=['temporal_behavior', 'network_selection', 'address_selection'],observed=True).size().unstack(fill_value=0)

    # Vollständige Liste aller möglichen Kombinationen erstellen
    temporal_levels = df['temporal_behavior'].unique()  # Dynamisch aus dem DataFrame
    network_levels = df['network_selection'].unique()    # Dynamisch aus dem DataFrame
    #address_types = df['address_selection'].unique()
    address_types = address_order 


    # Erstellen Sie einen DataFrame, um alle möglichen Kombinationen zu berücksichtigen
    index = pd.MultiIndex.from_product([temporal_levels, network_levels], names=['temporal_behavior', 'network_selection'])
    columns = address_types
    full_df = pd.DataFrame(index=index, columns=columns).fillna(0)

    # Fügen Sie die gezählten Daten in den DataFrame ein
    full_df.update(grouped)

    # Benutzerdefinierte Farben für das Balkendiagramm
    colors = ["#FB6542", "#c1195c", "#375E97"]

    sns.set(context="talk")
    nxplots = len(temporal_levels)  # Anzahl der Zeilen in der Subplot-Matrix
    nyplots = len(network_levels)    # Anzahl der Spalten in der Subplot-Matrix
    fig, axes = plt.subplots(nxplots, nyplots, sharey=True, sharex=True, figsize=fig_size)

    # Plotting the data
    for a, temporal in enumerate(temporal_levels):
        for i, network in enumerate(network_levels):
            # Extract data for the current combination
            data = full_df.loc[(temporal, network)]
            
            # Plot the bar chart
            bars = axes[a, i].bar(address_types, data, color=colors[:len(data)])
            
            # Logarithmic y-axis (check if positive values are present)
            if (data > 0).any():
                axes[a, i].set_yscale('log')
                axes[a, i].set_ylim(1, 1000000)

                # Set custom ticks and labels
                ticks = [10, 100, 1000, 10000, 100000]
                axes[a, i].set_yticks(ticks)

                # Set the y-axis tick label font size
                axes[a, i].tick_params(axis='y', labelsize=9)  # Set y-axis tick label font size

            # Remove titles for individual subplots
            axes[a, i].set_title('')
            # Remove x-axis ticks (optional, depending on desired layout)
            axes[a, i].xaxis.set_ticks([])

            # Ensure all y-tick labels have the same size
            for tick in axes[a, i].get_yticklabels():
                tick.set_fontsize(9)

            # Set x-axis label font size
            #axes[a, i].tick_params(axis='x', labelsize=9)  # Set x-axis tick label font size

    # Add telescope labels above the entire subplot grid
    for i, network in enumerate(network_levels):
        fig.text(
            0.21 + 0.205 * i,  # Adjust the positioning for better centering
            0.88,             # Position just above the plots (between 0 and 1)
            network,        # The telescope label
            ha='center',      # Centered horizontally
            va='top',         # Align at the top vertically
            fontsize=9       # Adjust the font size
        )

    # Set axis labels for the array of Axes
    for a, temporal in enumerate(temporal_levels):
        fig.text(
            0.92,                      # X-position (a little to the left of the subplots)
            0.74 - 0.26 * a,           # Y-position for each label (adjusted for each row)
            temporal,                  # The label (temporal behavior)
            ha='center',               # Center horizontally
            va='center',               # Center vertically
            rotation=-90,              # Rotate the text to be vertical
            fontsize=9                # Adjust font size
        )

    # Create space for the legend
    fig.subplots_adjust(top=0.85)

    # Add the legend and place it below the plot
    patches = [Patch(color=colors[i], label=address_order[i]) for i in range(len(address_order))]
    legend = fig.legend(
        handles=patches,
        loc="lower center",               # Set legend to lower center
        bbox_to_anchor=(0.5, 0.04),      # Position the legend below the plot
        ncol=len(address_order),          # Number of columns in the legend
        frameon=False,                    # Remove the legend frame
        prop={'size': 9}                 # Set font size for legend text
    )

    # Set marker size to make the patches smaller
    for patch in legend.get_patches():
        patch.set_height(10)  # Adjust the height of legend patches
        patch.set_width(20)   # Adjust the width of legend patches

    fig.text(0.02, 0.48, 'Sessions [#]', va='center', rotation='vertical', fontsize=9)
    fig.text(0.42, 0.04, 'Address selection', va='center', fontsize=9)
    fig.text(0.5, 0.916, 'Network selection', ha='center', va='top', fontsize=9)
    fig.text(0.96, 0.48, 'Temporal behavior', ha='center', va='center', rotation=-90, fontsize=9)

    plt.rc("font", size=9)
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)
    sns.reset_defaults()
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

@timeit
def sessions_presplit_per_telescope(dfs,labels,fig_name,fig_size,resample_interval,label_column,columns):
    df = presplit_filter(concat_frames(dfs, labels, label_column, columns))

    pickle_path = os.path.join(INTERIM_DATA_DIR, f"{fig_name}_cached.pkl")

    def process_df(df,resample_interval):
        return get_sessions_presplit(df,resample_interval)

    df_presplit_sessions128 = load_or_process_pickle(pickle_path, process_df, df,resample_interval)

    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)

    colors = ["#375E97", "#FB6542", "#c1195c", "#37975e","#a742f5","#42f5e6"]
    linestyles = ['--','solid','-.','dotted']
    markers = ['o','+','^','x']
    for i,source in enumerate(labels):
        ax.plot(df_presplit_sessions128.filter(pl.col('source')==source)['Timestamp'],df_presplit_sessions128.filter(pl.col('source')==source)['size'],label=f'{source.upper()}',marker=markers[i])#,linestyle=linestyles[0],color=colors[0])

    ax.set_yscale('log')
    ax.set_ylim(1,10**4.5)
    #ax.set_xlim('2023-08-24', '2023-11-21')
    ax.set_xlabel(f'Time [{resample_interval[-1].upper()}]')
    ax.set_ylabel('Sessions [#]')
    #ax.grid(axis='y')

    #ax.set_yticks([0,0.5,1])
    minorticks= pd.date_range('2023-08-24', '2023-11-21',freq='W')
    ax.set_xticks(minorticks, minor=True)

    plt.setp(ax, xticks=[pd.to_datetime('2023-09-01'),pd.to_datetime('2023-10-01'),pd.to_datetime('2023-11-01')], xticklabels=['Sep','Oct','Nov'])
    plt.xticks(rotation=0,ha='center')

    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    #ax.axvspan('2023-08-24', '2023-09-07', color='red', alpha=0.3)

    #handles, labels = ax.get_legend_handles_labels()
    #for handle in handles:
    #    handle.set_linewidth(2.0)
    #leg = ax.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5,1.4),columnspacing=1,ncol=3,fontsize=9,handlelength=2.5)

    # change the line width for the legend
    #for line in leg.get_lines():
    #    line.set_linewidth(2.5)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.4),ncols=4)

    fig.tight_layout()
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def sessions_per_prefix(df,fig_name,fig_size,columns,vertical_dates):
    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)
    grouped = df.select(columns).unique(subset=['Session_ID_128','MostSpecificPrefix'], keep='first').sort('Timestamp').collect().group_by('MostSpecificPrefix')
    #tmp = t1.drop_duplicates(['Session_ID_128','MostSpecificPrefix'],keep='first')
    # Define a custom sorting function
    def custom_sort_key(group):
        return group[0][0].split('/')[-1]

    # Sort the groups based on the custom key
    sorted_groups = sorted(grouped, key=custom_sort_key, reverse=False)

    prefixes = [prefix[0].split('/')[-1] for prefix,_ in sorted_groups]

    colors = ['#a6cee3','#1f78b4','#b2df8a',
            '#33a02c','#fb9a99','#e31a1c',
            '#fdbf6f','#ff7f00','#9e9ac8',
            '#3f007d','#a6ce00','#b15928',
            '#9e0142','#3288bd','#c994c7',
            '#006d2c','#e6550d','#810f7c']
    linestyles = ['--','solid','-.']

    p_id = 0
    for prefix,group in sorted_groups:
        if p_id==16:
            linestyle='--'
        else:
            linestyle='solid'
        prefix_data = group.group_by_dynamic('Timestamp',every='1d').agg(pl.len().alias('numsessions')).select([
        "Timestamp",
        pl.col("numsessions").cum_sum().alias("cumsumsessions")
    ])
        ax.plot(prefix_data['Timestamp'],prefix_data['cumsumsessions'],label=f'/{prefix[0].split("/")[-1]}',c=colors[p_id],linestyle=linestyle)
        p_id+=1

    y=16750

    for date,prefix in zip(vertical_dates,prefixes[:-1]):
        text = f'/{prefix}'
        ax.axvline(x=date, color='gray', linestyle='--',alpha=0.3)
        # timedelta for some space between grey line and text
        ax.text(x=date-pd.Timedelta(12,'h'), y=y,s=text,color='gray',rotation=90,horizontalalignment='right',fontsize=12,bbox=dict(facecolor='white', edgecolor='None', pad=0))

    ax.axvline(x=pd.to_datetime('2023-08-26'), color='gray', linestyle='--',alpha=0.5)
    ax.text(x=pd.to_datetime('2023-08-29'), y=y,s=f'/{32}',color='gray',rotation=90,horizontalalignment='left',fontsize=12,bbox=dict(facecolor='white', edgecolor='None', pad=0))

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(flip(handles, 17), flip(labels, 17), ncols=17,loc='upper center',bbox_to_anchor=(0.5,1.25),handlelength=1.1,columnspacing=1)
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(2)
        
    ax.set_ylabel('Sessions [#]',fontsize=12)
    ax.set_xlabel('Time [W]',fontsize=12)
    ax.set_xlim(pd.to_datetime('2023-08-24'),pd.to_datetime('2024-07-02'))
    ax.set_ylim(0,20000)
    #ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=0,ha='center')

    ax.set_yticks([0,5000,10000,15000,20000],minor=False)
    ax.set_yticklabels([f'{int(y)}k' if y>0 else '0' for y in ax.get_yticks()/1000])

    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    fig.tight_layout()
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def sources_sessions_t1_vs_other(dfs,labels,fig_name,fig_size,resample_interval,label_column,columns,vertical_dates):
    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)
    aggregated_df= concat_frames(dfs, labels, label_column, columns)

    other = aggregated_df.filter(pl.col(label_column)!='t1')
    t1_2we = dfs[0].sort('Timestamp').group_by_dynamic('Timestamp', every=resample_interval).agg(
        pl.col('scan_source_128').n_unique().alias('Unique Sources'),
        pl.col('Session_ID_128').n_unique().alias('Unique Sessions')
    ).collect().to_pandas()

    other_2we = other.sort('Timestamp').group_by_dynamic('Timestamp', every=resample_interval).agg(
        pl.col('scan_source_128').n_unique().alias('Unique Sources'),
        pl.col('Session_ID_128').n_unique().alias('Unique Sessions')
    ).collect().to_pandas()

    colors = ["#375E97", "#FB6542", "#c1195c", "#37975e"]
    linestyles = ['--','solid','-.',':']

    for date in vertical_dates:
        ax.axvline(x=date, color='gray', linestyle='--',alpha=0.3)

    ax.plot(other_2we['Timestamp'],other_2we['Unique Sources'],marker='^',markersize=4,linestyle='solid',color=colors[1],label='/128 Sources (Other)')
    ax.plot(other_2we['Timestamp'],other_2we['Unique Sessions'],marker='^',markersize=4,linestyle='--',color=colors[1],label='Sessions (Other)')
    ax.plot(t1_2we['Timestamp'],t1_2we['Unique Sources'],marker='s',markersize=4,linestyle='solid',color=colors[0],label='/128 Sources (T1)')
    ax.plot(t1_2we['Timestamp'],t1_2we['Unique Sessions'],marker='s',markersize=4,linestyle='--',color=colors[0],label='Sessions (T1)')

    handles, labels = ax.get_legend_handles_labels()

    legend = ax.legend(handles[::-1],labels[::-1],loc='upper right',bbox_to_anchor=(1,1.6,0,0),ncol=2,fontsize=11)

    ax.set_yscale('symlog')

    ax.set_ylabel('Observed [#]',fontsize=12)
    ax.set_xlabel('Time [2W]',fontsize=12)
    ax.set_xlim(t1_2we.Timestamp.min(),pd.to_datetime('2024-07-02'))
    ax.set_ylim(10**2,10**5)

    #ax.set_yticks([10**2,10**3,10**4,10**5],minor=False)
    #ax.set_yticklabels([f'{int(y)}K' if y>0 else '0' for y in ax.get_yticks()/1000])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=0,ha='center')

    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # change the line width for the legend
    #for line in legend.get_lines():
    #    line.set_linewidth(2.5)
        
    fig.tight_layout()

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def subnet_coverage_plot(df,fig_name,fig_size,columns):
    plt.rc("font", size=12)
    fig,ax = fig_ax(fig_size)

    pickle_path = os.path.join(INTERIM_DATA_DIR, f"{fig_name}_cached.pkl")

    def process_df(df,columns):
        return process_subnet_coverage_plot_data(df,columns)

    sorted_oneoff_df, sorted_periodic_df, sorted_intermittent_df = load_or_process_pickle(pickle_path,process_df,df,columns)

    colors = ["#375E97", "#FB6542", "#c1195c", "#37975e"]
    # Erstellen der Step-Plots für jede Kategorie mit benutzerdefinierten Farben
    plt.step(sorted_oneoff_df.index, sorted_oneoff_df['count_dst'], label='One-off', color=colors[0])
    #plt.step(sorted_triggered_periodic_df.index, sorted_triggered_periodic_df['count_dst'], label='Triggered and periodic', color=colors[1])
    plt.step(sorted_periodic_df.index, sorted_periodic_df['count_dst'], label='Periodic', color=colors[2])
    plt.step(sorted_intermittent_df.index, sorted_intermittent_df['count_dst'], label='Intermittent', color=colors[3])

    # Konfiguration der Legende
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize='medium', columnspacing=1, handletextpad=0.3)

    # Konfiguration des Plots
    plt.xlabel('/48 subnet ranked per packet reception')
    plt.ylabel('Packets [#]')
    plt.yscale("log")

    # Anpassen der Y-Achse
    ax.set_ylim(2, 5000000)
    ax.set_yticks([100,10000,1000000])
    ax.set_yticklabels(['$10^2$', '$10^4$', '$10^6$'])
    
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def nist_plot(fig_name,fig_size):
    oneoff_subnet_df = pd.read_parquet(NIST_ONEOFF_SUBNET)
    periodic_subnet_df = pd.read_parquet(NIST_PERIODIC_SUBNET)
    intermittent_subnet_df = pd.read_parquet(NIST_INTERMITTENT_SUBNET)

    oneoff_iid_df = pd.read_parquet(NIST_ONEOFF_IID)
    periodic_iid_df = pd.read_parquet(NIST_PERIODIC_IID)
    intermittent_iid_df = pd.read_parquet(NIST_INTERMITTENT_IID)

    oneoff_subnet_df['Experiment'] = oneoff_subnet_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    oneoff_subnet_df['Experiment'] = oneoff_subnet_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    #triggered_periodic_subnet_df['Experiment'] = triggered_periodic_subnet_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    #triggered_periodic_subnet_df['Experiment'] = triggered_periodic_subnet_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    periodic_subnet_df['Experiment'] = periodic_subnet_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    periodic_subnet_df['Experiment'] = periodic_subnet_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    intermittent_subnet_df['Experiment'] = intermittent_subnet_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    intermittent_subnet_df['Experiment'] = intermittent_subnet_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    oneoff_iid_df['Experiment'] = oneoff_iid_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    oneoff_iid_df['Experiment'] = oneoff_iid_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    #triggered_periodic_iid_df['Experiment'] = triggered_periodic_iid_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    #triggered_periodic_iid_df['Experiment'] = triggered_periodic_iid_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    periodic_iid_df['Experiment'] = periodic_iid_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    periodic_iid_df['Experiment'] = periodic_iid_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    intermittent_iid_df['Experiment'] = intermittent_iid_df['Experiment'].replace({'CumulativeSums1': 'Cusum0'})
    intermittent_iid_df['Experiment'] = intermittent_iid_df['Experiment'].replace({'CumulativeSums2': 'Cusum1'})

    # Funktion zur Erstellung eines Balkendiagramms für ein DataFrame
    def create_bar_chart(ax, df):
        # Gruppiere den DataFrame nach der Spalte "Experiment" und summiere die Werte in der Spalte "Status"
        grouped_df = df.groupby('Experiment')['Status'].value_counts().unstack(fill_value=0)

        # Definiere die gewünschte Reihenfolge der Kategorien
        category_order = ['Frequency', 'Runs', 'FFT', 'Cusum0', 'Cusum1']

        # Reindexiere grouped_df basierend auf der gewünschten Reihenfolge
        grouped_df = grouped_df.reindex(category_order, axis=0)

        # Berechne die Gesamtanzahl der Einträge pro Experiment
        grouped_df['total'] = grouped_df.sum(axis=1)

        # Prozentsatz von 'pass' und 'failed' für jedes Experiment berechnen
        grouped_df['pass_percentage'] = (grouped_df['pass'] / grouped_df['total']) * 100
        grouped_df['failed_percentage'] = (grouped_df['failed'] / grouped_df['total']) * 100

        width = 0.2  # Breite der Balken
        x = range(len(grouped_df))

        # Manuell festgelegte Farben für die Balken
        failed_color = '#375E97'
        pass_color = '#FB6542'

        # Erstelle die Balken für "failed" und "pass" mit den Prozentsätzen
        failed_bars = ax.bar(x, grouped_df['failed_percentage'], width, label='failure', color=failed_color)
        pass_bars = ax.bar([i + width for i in x], grouped_df['pass_percentage'], width, label='success', color=pass_color)

        # Achsen- und Beschriftungseinstellungen
        #ax.set_xlabel('Randomness Test [#]')
        
        #ax.set_ylim(0, 100)  # y-Achse von 0% bis 100%
        ax.set_ylim(0, 100)[2::]
        ax.set_xticks(ax.get_xticks()[::2])
        ax.tick_params(axis='x', labelsize=7)

        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(grouped_df.index)
        ax.tick_params(axis='y', labelsize=7)

    # Erstelle Balkendiagramme für jedes DataFrame in einer 4x2-Kachelanordnung
    fig, axes = fig_ax(fig_size, sharex=True, sharey=True, nrows=3, ncols=2)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Reduziere den horizontalen und vertikalen Abstand zwischen den Subplots
    plt.rc("font", size=9)

    # Liste der DataFrames und zugehörigen Titel
    dataframes = [oneoff_iid_df, oneoff_subnet_df, periodic_iid_df, periodic_subnet_df, intermittent_iid_df, intermittent_subnet_df]
    titles = ['One-off', 'One-off', 'Periodic', 'Periodic', 'Intermittent', 'Intermittent']

    i=0
    # Schleife über die Subplots und DataFrames
    for ax, df, title in zip(axes.flat, dataframes, titles):
        create_bar_chart(ax, df)
        if i%2==0:
            ax.set_ylabel('Share [%]')
        i+=1
        ax.set_title(title, fontsize=7)  # Setze den individuellen Titel für jeden Subplot

    # Legende nur einmal anzeigen
    axes[0, 0].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.38, 1), fontsize=7)

    # Achsenbeschriftungen
    #fig.text(0.06, 0.5, 'Share [%]', va='center', rotation='vertical',fontsize=9)
    fig.text(0.45, 0.05, 'NIST Test', va='center', fontsize=9)

    # Zusätzliche Anmerkungen für die Spaltenüberschriften
    fig.text(0.205, 0.95, 'Interface identifier', va='center', fontsize=9)
    fig.text(0.68, 0.95, 'Subnet', va='center', fontsize=9)
    #fig.text(0.28, 0.5, 'oneoff', va='center', fontsize=12)
    #fig.text(0.669, 0, 'intermittent', va='center', fontsize=12)
    #fig.text(0.28, 0, 'periodic', va='center', fontsize=12)
    
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def upsetplot_for_column_filtered(dfs, labels, column,label,title,sfx='',bbox=(0.5,1),minsubsetsize=0):
    suffix = f'{column}_{sfx}'
    columns=[column,'Date']

    tmp = presplit_filter(concat_frames(dfs,labels,'source', columns),'Date').collect().to_pandas()
    tmp.drop('Date',axis=1,inplace=True)

    plot_upsetplot(tmp,column,'source',labels,f'upsetplot_telescopes_{suffix}',
                   label,title = title,bbox=bbox,pickle_file=f'{INTERIM_DATA_DIR}/upsetplot_telescopes_{column}.pkl',minsubsetsize=minsubsetsize)

@timeit
def overlap_cumulative_first_observation_plot(dfs,fig_name,fig_size,vertical_dates):
    plt.rc("font", size=12)
    plt.rc("figure", figsize=fig_size)

    start_date=datetime(2023, 8, 24)
    end_date=datetime(2024,7,2)
    dft1 = dfs[0].collect()
    dft2 = dfs[1].collect()
    addrs_t1 = set(pl.Series(dft1.select(pl.col("Source_Address"))).unique().to_list())
    addrs_t2 = set(pl.Series(dft2.select(pl.col("Source_Address"))).unique().to_list())
    t1_t2 = addrs_t1.intersection(addrs_t2)
    ## Figure 16b
    def filter_t1_and_t2(df):
        return df.filter(pl.col("Source_Address").is_in(t1_t2)).select(
            pl.col("Timestamp"),
            pl.col("Source_Address"),
            pl.col("Protocol"),
        ).with_columns(
            pl.col("Timestamp").dt.date().alias("Date"),
        ).group_by("Source_Address").agg(
            pl.col("Date").min().alias("First"),
            pl.col("Date").max().alias("Last"),
            pl.col("Date").n_unique().alias("Days")
        ).with_columns(
            ((pl.col("Last") - pl.col("First")).dt.total_days() + 1).alias("Duration")
        )


    overlap_t1_t1t2 = filter_t1_and_t2(dft1)
    overlap_t2_t1t2 = filter_t1_and_t2(dft2)

    dates_abs_first_df = overlap_t1_t1t2.with_columns(
        pl.col("First").alias("First T1"),
        pl.col("Last").alias("Last T1"),
    ).select(
        pl.col("Source_Address"),
        pl.col("First T1"),
        pl.col("Last T1"),
    ).join(overlap_t2_t1t2.with_columns(
        pl.col("First").alias("First T2"),
        pl.col("Last").alias("Last T2"),
    ).select(
        pl.col("Source_Address"),
        pl.col("First T2"),
        pl.col("Last T2"),
    ), on="Source_Address").with_columns(
        (abs(pl.col("First T1") - pl.col("First T2"))).dt.total_days().alias("First Diff"),
        (abs(pl.col("Last T1") - pl.col("Last T2"))).dt.total_days().alias("Last Diff")
    )

    threshold = 0

    pos_label = f"Same day" # f"Meets Threshold"
    neg_label = f"Different day" # f"Fails Threshold"

    plot_df = dates_abs_first_df.with_columns(
        pl.max_horizontal(["First T1", "First T2"]).alias("First"),
        # pl.min_horizontal(["First T1", "First T2"]).alias("First"),
        (pl.when(pl.col('First Diff')<=threshold)
            .then(1)
            .otherwise(0)).alias(pos_label),
        (pl.when(pl.col('First Diff') > threshold)
            .then(1)
            .otherwise(0)).alias(neg_label),
    ).sort("First").group_by("First").agg(
        pl.col(pos_label).sum().alias(pos_label),
        pl.col(neg_label).sum().alias(neg_label),
    ).to_pandas()

    # start_date = datetime(2023,9,8)

    range = pd.date_range(start=start_date, end=end_date, freq='D')
    plot_df = plot_df.set_index("First").reindex(index=range)
    plot_df["Total"] = plot_df[pos_label] + plot_df[neg_label]

    plot_df = plot_df.fillna(0)

    plot_df["Total"] = plot_df["Total"].cumsum()
    plot_df[pos_label] = plot_df[pos_label].cumsum()
    plot_df[neg_label] = plot_df[neg_label].cumsum()

    plot_df[pos_label] = plot_df[pos_label] / plot_df["Total"] * 100
    plot_df[neg_label] = plot_df[neg_label] / plot_df["Total"] * 100

    plot_df = plot_df.drop("Total", axis=1)

    ax = plot_df.plot.area()

    ax.set_ylim([0, 100])
    ax.axvline(vertical_dates[0], color="grey", linestyle="--", linewidth=1)

    # ax.axhline(50, color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel("Time [D]")

    ax.set_ylabel("Share [%]")

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    # ax.legend(bbox_to_anchor=(0.5, 1.15), handles=legend_elements, loc='center', ncol=4, fontsize=11)
    ax.legend(loc='lower center', ncol=2) #, ncol=4, fontsize=11)
    fig = plt.gcf()

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def overlap_addresses_all_telescopes(dfs, fig_name, fig_size, vertical_dates):
    dft1 = dfs[0].collect()
    dft2 = dfs[1].collect()
    dft3 = dfs[2].collect()
    dft4 = dfs[3].collect()
    may_colors = ['#a6cee3','#1f78b4','#b2df8a',
          '#33a02c','#fb9a99','#e31a1c',
          '#fdbf6f','#ff7f00','#9e9ac8',
          '#3f007d','#a6cee3','#b15928',
         '#e0ecf4','#f7fcb9','#c994c7',
          '#006d2c','#e6550d','#810f7c']
    start_date=datetime(2023, 8, 24)
    end_date=datetime(2024,7,2)
    min_date=start_date
    max_date=end_date

    addrs_t1 = set(pl.Series(dft1.select(pl.col("Source_Address"))).unique().to_list())
    addrs_t2 = set(pl.Series(dft2.select(pl.col("Source_Address"))).unique().to_list())
    addrs_t3 = set(pl.Series(dft3.select(pl.col("Source_Address"))).unique().to_list())
    addrs_t4 = set(pl.Series(dft4.select(pl.col("Source_Address"))).unique().to_list())
    seen_by_all = addrs_t1.intersection(addrs_t2).intersection(addrs_t3).intersection(addrs_t4)
    seen_by_all_ordered = [
        '2001:253:ff:4301:3673:79ff:fe91:48b8',
        '2001:550:9005:1000::11',
        '2001:550:9005:2000::11',
        '2001:da8:bf:300:ae1f:6bff:fefb:8b62',
        '2401:c080:1c00:2d8c:5400:4ff:feb3:fd43',
        '2401:c080:1c02:d91:5400:4ff:feaa:f563',
        '2602:ffd5:1:164::1',
        '2605:6400:10:6a8:bc7a:408:1ad:e7e6',
        '2a01:190:151a:1::5ba:100',
        '2a01:190:151a::5ba:100'
    ]
    # -----

    def log_scale_sizes(counts, scale=20):
        counts = np.array(counts)
        return scale * np.log1p(counts)  # log1p to avoid log(0)

    fig, ax = fig_ax(fig_size)

    for pos, addr in enumerate(seen_by_all_ordered):

        filter_set = set([addr])

        def filter_all(df, name):
            return df.filter(pl.col("Source_Address").is_in(filter_set)).select(
                pl.col("Timestamp"),
                pl.col("Source_Address"),
                pl.col("Protocol"),
            ).with_columns(
                pl.col("Timestamp").dt.date().alias("Date"),
            ).group_by([pl.col("Date")]).agg(
                pl.col("Source_Address").count().alias(name)
            )

        overlap_t1 = filter_all(dft1, "T1")
        overlap_t2 = filter_all(dft2, "T2")
        overlap_t3 = filter_all(dft3, "T3")
        overlap_t4 = filter_all(dft4, "T4")
        sum_t1 = overlap_t1["T1"].sum()
        sum_t2 = overlap_t2["T2"].sum()
        sum_t3 = overlap_t3["T3"].sum()
        sum_t4 = overlap_t4["T4"].sum()

        total = sum_t1 + sum_t2 + sum_t3 + sum_t4

        # Create one df

        overlap_df = overlap_t1.join(overlap_t2, on="Date", how="full", coalesce=True)
        overlap_df = overlap_df.join(overlap_t3, on="Date", how="full", coalesce=True)
        overlap_df = overlap_df.join(overlap_t4, on="Date", how="full", coalesce=True)
        # overlap_df

        # Turn to pandas and reindex.
        plot_df = overlap_df.sort(pl.col("Date")).to_pandas()

        plot_df = plot_df.set_index("Date")
        range = pd.date_range(start=start_date, end=end_date, freq='D')
        plot_df = plot_df.reindex(index=range)

        telescopes = ["T1", "T2", "T3", "T4"]

        for i, scope in enumerate(telescopes):
            sizes = log_scale_sizes(plot_df[scope])
            # sizes = sqrt_scale_sizes(plot_df[scope])
            # sizes = custom_scale_sizes(plot_df[scope])
            xs = list(plot_df.index)
            ys = [i] * len(xs)

            ax.scatter(
                y=ys,
                x=xs,
                s=sizes,
                color=may_colors[pos],
                edgecolors='black',
                linewidths=0.1,
                # alpha=0.7,
            )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    # ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_ticks([0, 1, 2, 3])
    ax.yaxis.set_ticklabels(["T1", "T2", "T3", "T4"])

    # ax.set_ylim([10**0, 10**6])

    ax.set_xlim([start_date, end_date])
    ax.set_ylim([-0.5, 3.3])

    ax.axvline(vertical_dates[0], color="grey", linestyle="--", linewidth=1)

    ax.set_ylabel("Telescope")
    ax.set_xlabel("Time [D]")

    plt.gca().invert_yaxis()

    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

@timeit
def ponynet_random_scanner_heatmap(df,fig_name,fig_size,columns):
    ipv6_dest_2 = df.select(columns).filter((pl.col('Source_Address') == '2605:6400:10:6a8:bc7a:408:1ad:e7e6') & (pl.col('Session_ID_128')==44763)).sort(by='Timestamp').select('fullhex_destination_address').collect().to_pandas()
    
    ipv6_dest_2 = ipv6_dest_2.reset_index(drop=True)

    max_length = max(len(addr) for addr in ipv6_dest_2['fullhex_destination_address'])

    palette = sns.color_palette("YlGnBu", 16)
    heatmap_data = np.zeros((max_length, len(ipv6_dest_2)), dtype=int)

    for i, row in enumerate(ipv6_dest_2['fullhex_destination_address']):
        for j, char in enumerate(row):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.lower()) - ord('a') + 10
            else:
                continue 
            heatmap_data[j, i] = value

    plt.figure(figsize=fig_size)
    plt.rc("font", size=9)

    heatmap = sns.heatmap(heatmap_data, cmap=palette, cbar=True, cbar_kws={'label': 'Hexadecimal character'})

    plt.xlabel('Order of arrival [#]')
    plt.ylabel('IPv6 address [nibble]')

    plt.xticks([])
    major_locator = ticker.MultipleLocator(4)
    minor_locator = ticker.MultipleLocator(2)

    plt.gca().yaxis.set_major_locator(major_locator)

    y_ticks_positions = [3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5, 31.5]
    y_ticks_labels = [4, 8, 12, 16, 20, 24, 28, 32]
    plt.yticks(y_ticks_positions, labels=y_ticks_labels)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks(np.linspace(0, 15, 16))
    colorbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'])
    colorbar.ax.set_aspect(3)

    for i in range(heatmap_data.shape[1]):
        for j in range(8):
            heatmap.add_patch(Rectangle((i, j), 1, 1, fill=True, color='#D1D1D1'))

    plt.text(0.5, 0.875, 'Telescope prefix', va='center', ha='center', fontsize=10, color='#000000', transform=plt.gca().transAxes)

    fig = plt.gcf()
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True,dpi=150,pngonly=True)

@timeit
def tencent_structured_scanner_heatmap(df,fig_name,fig_size,columns):
    ipv6_dest_2 = df.select(columns).filter((pl.col('Source_Address') == '240d:c000:2010:1a42:0:98e7:da67:1fb7') & (pl.col('Session_ID_128')==33040)).sort(by='Timestamp').select('fullhex_destination_address').collect().to_pandas()
    
    ipv6_dest_2 = ipv6_dest_2.reset_index(drop=True)

    max_length = max(len(addr) for addr in ipv6_dest_2['fullhex_destination_address'])

    palette = sns.color_palette("YlGnBu", 16)
    heatmap_data = np.zeros((max_length, len(ipv6_dest_2)), dtype=int)

    for i, row in enumerate(ipv6_dest_2['fullhex_destination_address']):
        for j, char in enumerate(row):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.lower()) - ord('a') + 10
            else:
                continue 
            heatmap_data[j, i] = value

    plt.figure(figsize=fig_size)
    plt.rc("font", size=9)

    heatmap = sns.heatmap(heatmap_data, cmap=palette, cbar=True, cbar_kws={'label': 'Hexadecimal character'})

    plt.xlabel('Order of arrival [#]')
    plt.ylabel('IPv6 address [nibble]')

    plt.xticks([])
    major_locator = ticker.MultipleLocator(4)
    minor_locator = ticker.MultipleLocator(2)

    plt.gca().yaxis.set_major_locator(major_locator)

    y_ticks_positions = [3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5, 31.5]
    y_ticks_labels = [4, 8, 12, 16, 20, 24, 28, 32]
    plt.yticks(y_ticks_positions, labels=y_ticks_labels)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks(np.linspace(0, 15, 16))
    colorbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'])
    colorbar.ax.set_aspect(3)

    for i in range(heatmap_data.shape[1]):
        for j in range(8):
            heatmap.add_patch(Rectangle((i, j), 1, 1, fill=True, color='#D1D1D1'))

    plt.text(0.5, 0.875, 'Telescope prefix', va='center', ha='center', fontsize=10, color='#000000', transform=plt.gca().transAxes)

    fig = plt.gcf()
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True,dpi=150,pngonly=True)

@timeit
def tencent_numeric_ordered_heatmap(df,fig_name,fig_size,columns):
    ipv6_dest_2 = df.select(columns).filter((pl.col('Source_Address') == '240d:c000:2010:1a42:0:98e7:da67:1fb7') & (pl.col('Session_ID_128')==33040)).sort(by='fullhex_destination_address').select('fullhex_destination_address').collect().to_pandas()
    
    ipv6_dest_2 = ipv6_dest_2.reset_index(drop=True)

    max_length = max(len(addr) for addr in ipv6_dest_2['fullhex_destination_address'])

    palette = sns.color_palette("YlGnBu", 16)
    heatmap_data = np.zeros((max_length, len(ipv6_dest_2)), dtype=int)

    for i, row in enumerate(ipv6_dest_2['fullhex_destination_address']):
        for j, char in enumerate(row):
            if char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.lower()) - ord('a') + 10
            else:
                continue 
            heatmap_data[j, i] = value
    plt.figure(figsize=fig_size)
    plt.rc("font", size=9)

    heatmap = sns.heatmap(heatmap_data, cmap=palette, cbar=True, cbar_kws={'label': 'Hexadecimal character'})

    plt.xlabel('Order of address [#]')
    plt.ylabel('IPv6 address [nibble]')

    plt.xticks([])
    major_locator = ticker.MultipleLocator(4)
    minor_locator = ticker.MultipleLocator(2)

    plt.gca().yaxis.set_major_locator(major_locator)

    y_ticks_positions = [3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5, 31.5]
    y_ticks_labels = [4, 8, 12, 16, 20, 24, 28, 32]
    plt.yticks(y_ticks_positions, labels=y_ticks_labels)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks(np.linspace(0, 15, 16))
    colorbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'])
    colorbar.ax.set_aspect(3)

    for i in range(heatmap_data.shape[1]):
        for j in range(8):
            heatmap.add_patch(Rectangle((i, j), 1, 1, fill=True, color='#D1D1D1'))

    plt.text(0.5, 0.875, 'Telescope prefix', va='center', ha='center', fontsize=10, color='#000000', transform=plt.gca().transAxes)

    fig = plt.gcf()
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True,dpi=150,pngonly=True)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    logger.info("Reading data...")
    pl.enable_string_cache()
    t1 = pl.scan_parquet(T1_DATAFRAME) 
    t2 = pl.scan_parquet(T2_DATAFRAME) 
    t3 = pl.scan_parquet(T3_DATAFRAME) 
    t4 = pl.scan_parquet(T4_DATAFRAME)
    announcement_df = pd.read_csv(ANNOUNCEMENT_LOG_FILE)
    vertical_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in announcement_df[1:].Timestamp_From.unique()]

    logger.info("Generating plot from data...")
    logger.info("Generating Figure 04...")
    columns=['Timestamp','prefix_target']
    new_prefixes(t1,FIG_NEW_PREFIXES, FIGSIZE_SMALL,columns)

    logger.info("Generating Figure 05 -- Aggregated telescopes full period...")
    columns = ['Timestamp','Source_Address','Destination_Address','Protocol',
           'Geo','AS-Number','Org',
          'UDP_dst_port','TCP_dst_port','scan_source_128','scan_source_32',
          'scan_source_48','scan_source_64','Session_ID_128','Session_ID_64',
          'Session_ID_48','dest_addr_type']
    aggregated_telescopes_full_period([t1,t2,t3,t4],['t1','t2','t3','t4'],FIG_AGG_TELESCOPES_FULL, FIGSIZE_SMALL_WITH_LEGEND_ON_TOP,'1d', 'source', columns)

    logger.info("Generating Figure 06 -- Heavy hitter bubbles...")
    #columns=['Timestamp','prefix_target']
    heavy_hitter_bubble([t1,t2,t3,t4],['t1','t2','t3','t4'],FIG_HEAVY_HITTER_BUBBLES, FIGSIZE_SMALL,columns,vertical_dates)

    logger.info("Generating Figure 08a...")
    columns=['Timestamp','prefix_target']
    network_traffic_overview([t1,t2,t3,t4],['T1 (BGP controlled)','T2 (Partially productive)','T3 (Silent)','T4 (Reactive)'],FIG_NETWORK_TRAFFIC_OVERVIEW, FIGSIZE_LONG,'1h')

    logger.info("Generating Figure 08b...")
    tmp_df = pd.read_csv(TAXONOMY_BEFORE_SPLIT)
    taxonomy_plot_before_split(tmp_df,FIG_TAX_BEFORE_SPLIT, FIGSIZE_LONG)

    logger.info("Generating Figure 16...")
    tmp_df = pd.read_csv(TAXONOMY_DURING_SPLIT)
    taxonomy_plot_during_split(tmp_df,FIG_TAX_DURING_SPLIT, FIGSIZE_LONG)

    logger.info("Generating Figure 09a and 09b...")
    upsetplot_for_column_filtered([t1,t2,t3,t4],['T1','T2','T3','T4'],'AS-Number','ASN','','presplit_longversion',bbox=(1,1),minsubsetsize=0)
    upsetplot_for_column_filtered([t1,t2,t3,t4],['T1','T2','T3','T4'],'scan_source_128','/128','','presplit_longversion',bbox=(1,1),minsubsetsize=0)

    logger.info("Generating Figure 10...")
    columns=['Timestamp','Session_ID_128']
    sessions_presplit_per_telescope([t1,t2,t3,t4],['t1','t2','t3','t4'],FIG_SESSIONS_PRESPLIT,FIGSIZE_SPECIAL,'1w','source',columns)

    logger.info("Generating Figure 11...")
    columns=['Timestamp','MostSpecificPrefix','Session_ID_128']
    sessions_per_prefix(t1,FIG_SESSIONS_PER_PREFIX, FIGSIZE_EXTRA_WIDE,columns,vertical_dates)

    logger.info("Generating Figure 12...")
    columns=['Timestamp','scan_source_128','Session_ID_128']
    sources_sessions_t1_vs_other([t1,t2,t3,t4],['t1','t2','t3','t4'],FIG_SRC_SES_T1_VS_OTHER,FIGSIZE_SMALL_2,'2w','source',columns,vertical_dates)

    logger.info("Generating Figure 15...")
    columns=['Timestamp', 'Source_Address', 'Destination_Address', 'fullhex_destination_address', 'Session_ID_128', 'is_oneoff_128', 'dest_addr_type', 'period_128']
    subnet_coverage_plot(t1,FIG_SUBNET_COVERAGE, FIGSIZE_SMALL_WITH_LEGEND_ON_TOP,columns)
    
    logger.info("Generating Figure 17a...")
    overlap_addresses_all_telescopes([t1,t2,t3,t4],FIG_OVERLAP_ADDR_ALL,FIGSIZE_SMALL,vertical_dates)
    logger.info("Generating Figure 17b...")
    overlap_cumulative_first_observation_plot([t1,t2],FIG_OVERLAP_CUM,FIGSIZE_SMALL,vertical_dates)

    logger.info("Generating Figure 18...")
    nist_plot(FIG_NIST,FIGSIZE_LONG)

    logger.info('Rendering lots of data...go and get a coffee, this may take some time.')
    logger.info("Generating Figure 13a...")
    columns = ['Timestamp','Source_Address','Session_ID_128','fullhex_destination_address']
    tencent_structured_scanner_heatmap(t1,FIG_TENCENT_STRUCTURED,FIGSIZE_SMALL_3,columns)
    
    logger.info("Generating Figure 13b...")
    columns = ['Timestamp','Source_Address','Session_ID_128','fullhex_destination_address']
    ponynet_random_scanner_heatmap(t1,FIG_PONYNET_RANDOM,FIGSIZE_SMALL_3,columns)

    logger.info("Generating Figure 14...")
    columns = ['Timestamp','Source_Address','Session_ID_128','fullhex_destination_address']
    tencent_numeric_ordered_heatmap(t1,FIG_TENCENT_SORTED,FIGSIZE_SMALL_3,columns)

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
