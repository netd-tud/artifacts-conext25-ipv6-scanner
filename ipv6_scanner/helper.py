from pathlib import Path
from enum import StrEnum

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple

from ipv6_scanner.config import FIGURES_DIR

from tqdm import tqdm

import os
import pickle
import itertools

import pandas as pd

def fig_ax(figsize: tuple, **kwargs) -> Tuple[Figure, Axes]:
    """Wrapper for plt.subplots with default figure size.

    Args:
        figsize: figsize for plt.sublots, default chosen based on paper kind
        **kwargs: kwargs for plt.subplots

    Returns:
        tuple[Figure, Axes]: similar to plt.subplots
    """
    return plt.subplots(figsize=figsize, **kwargs)


def save_plot(
    fig: Figure, file_name: str, directory: Path = FIGURES_DIR, autoclose: bool = False
):
    """Save figure into FIGURES_DIR in high resolution png and pdf format.

    Args:
        fig: matplotlib figure
        file_name: filename without file extensions. Suggested pattern: f'{3ltr_plot_kind}_{description}'
        directory: destination folder, default: FIGURES_DIR
        autoclose: autoclose matplotlib figures
    """
    if file_name is not None:
        print(directory / f"{file_name}.png")
        fig.savefig(directory / f"{file_name}.png", bbox_inches="tight", dpi=200)
        fig.savefig(directory / f"{file_name}.pdf", bbox_inches="tight")

    if autoclose:
        plt.close()


def load_or_process_pickle(pickle_path, process_func, *args, **kwargs):
    """
    Load a DataFrame from a pickle file if it exists,
    otherwise run the processing function, save the result, and return it.

    Args:
        pickle_path: Path to the pickle file.
        process_func: Function to generate the DataFrame.
        *args, **kwargs: Arguments passed to process_func.

    Returns:
        DataFrame: The loaded or processed DataFrame.
    """
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        result = process_func(*args, **kwargs)
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        return result

def flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

def plot_upsetplot(df,groupbycol,setcol,tags,fig_name,label,colors = ["#375E97", "#FB6542", "#c1195c", "#37975e",'#cc9900'],title='Upsetplot',bbox=(0.5,1),minsubsetsize=0.01,pickle_file=None):
    import upsetplot
    import itertools
    from matplotlib.ticker import FuncFormatter
    import matplotlib.patches as mpatches
    import warnings
    import os
    warnings.filterwarnings('ignore')
    def format_tick_label(label, pos):
        if label >= 1000000:
            return f"{label/1000000:.1f}M"
        elif label >= 1000:
            return f"{label/1000:.0f}k"
        else:
            return f"{label:.0f}"

    def check_exclusive_entries(df, columns2check,groupbycolumn,setcolumn):
        missing_entries = []
        for col in columns2check:
            if not ((df[col] == 1) & (df[columns2check].sum(axis=1) == 1)).any():
                new_row = {groupbycolumn: f'X{len(missing_entries)+1}', setcolumn: {col}}
                new_row.update({c: 1 if c == col else 0 for c in columns2check})
                missing_entries.append(new_row)
        return missing_entries

    def count_ones(tup):
        return sum(x == 1 for x in tup)

    pbar = tqdm(total=10)
    plt.rcParams["font.size"] = 8
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.shadow"] = False
    pbar.set_description('Creating grouped sets')
    if pickle_file and os.path.exists(pickle_file):
        pbar.set_description('Found pickle file...loading')
        combined_df = pd.read_pickle(pickle_file)
        pbar.update(4)
    else:
        grouped = df.groupby(groupbycol,observed=True)[setcol].apply(set)
        pbar.update(1)
        # Convert the grouped Series to a DataFrame
        df_sets = grouped.reset_index()
        # Convert the sets into binary indicators for each possible TransportLayer
        pbar.set_description('Creating binary layers')
        for layer in df[setcol].unique():
            df_sets[layer] = df_sets[setcol].apply(lambda x: 1 if layer in x else 0)
        pbar.update(1)
        pbar.set_description('Add missing rows to dataframe')
        # Add missing rows to the DataFrame
        missing_rows = check_exclusive_entries(df_sets, tags,groupbycol,setcol)
        #if missing_rows:
        #    df_sets = pd.concat([df_sets, pd.DataFrame(missing_rows)], ignore_index=True)
    
        # Remove the original TransportLayer column
        df_sets.drop(columns=[setcol], inplace=True)
        #print(df_sets)
        combined_df = df_sets.groupby(tags).size()
        
        # Generate all possible combinations of T1, T2, T3, T4 (binary combinations)
        all_combinations = list(itertools.product([0, 1], repeat=len(tags)))
        #print(all_combinations)
    
        # Find missing combinations by comparing with the current index
        existing_combinations = set(combined_df.index)
        #print(existing_combinations)
        missing_combinations = [combo for combo in all_combinations if combo not in existing_combinations and any(combo)]
        #print(missing_combinations)
        # Create a DataFrame for the missing combinations with Value set to 0
        missing_df = pd.Series(0, index=pd.MultiIndex.from_tuples(missing_combinations, names=tags))
        # Concatenate the existing dataframe with the missing combinations dataframe
        combined_df = pd.concat([combined_df, missing_df])
        pbar.update(1)
        pbar.set_description('Sorting index')
        # Sort the series by the number of 1s (primary) and by their inversed lexicographical order
        # --> as a result we get (1,0,0,0), (0,1,0,0), (0,0,1,0)...
        sorted_index = sorted(combined_df.index, key=lambda x: (count_ones(x), tuple(reversed(x))))
        combined_df = combined_df.reindex(sorted_index)
        #print(combined_df)
        # Create the upset plot
        pbar.update(1)
        if pickle_file:
            combined_df.to_pickle(pickle_file)
            pbar.set_description('Creating pickle file for later runs')
    if minsubsetsize>0:
        total = combined_df.sum()
        keep = []
        for index in combined_df.index:
            if sum(index)==1:
                keep.append(index)
                continue
            if sum(index)>1:
                v = combined_df[combined_df.index==index].array[0]
                if v/total >= minsubsetsize:
                    keep.append(index)
        combined_df = combined_df[combined_df.index.isin(keep)]
    pbar.set_description('Creating UpSetPlot')
    upset = upsetplot.UpSet(
        #upsetplot.from_indicators(indicators=pd.isna,data=combined_df),
        combined_df,
        sort_categories_by = '-input',
        sort_by = 'input',
        # set_counts_log
        # orientation="vertical",
        # show_counts=True,
        show_percentages="{:.1%}",  # True,
    )
    pbar.update(1)
    pbar.set_description('Styling UpSetPlot')
    # config
    lw = 2
    for i in range(1, len(tags) + 1):
        for tag in itertools.combinations(tags, r=i):
            upset.style_subsets(
                present=tag,
                facecolor=colors[0],
                label="Overlap",
                linewidth=lw,
            )
            
    for i,tag in enumerate(tags,start=1):
        upset.style_subsets(
            present=tag,
            absent=[t for t in tags if t not in [tag]],
            facecolor=colors[i],
            label=tag,
            linewidth=lw,
        )
    pbar.update(1)
    pbar.set_description('Plotting UpSetPlot')
    plot_result = upset.plot()#.drop(columns=['Session_ID_128']))
    pbar.update(1)
    pbar.set_description('Fine-tuning of UpSetPlot')
    params_size = 11
    label_size = 11
    # Intersection plot configuration (the one with the overlapping sizes)
    # offset = plot_result["intersections"].yaxis.get_major_formatter().get_offset()
    plot_result["intersections"].set_ylabel(f"{label} Intersections [#]")
    plot_result["intersections"].yaxis.label.set_size(label_size)
    plot_result["intersections"].tick_params(axis="y", which="major", labelsize=params_size)
    # Remove 1e6 label (or move it to the bottom?)
    # https://stackoverflow.com/a/43719468/21257387
    plot_result["intersections"].yaxis.offsetText.set_visible(False)
    plot_result["intersections"].yaxis.set_major_formatter(FuncFormatter(format_tick_label))
    
    # Totals plot configuration (The few ones)
    plot_result["totals"].set_xlabel(f"{label} Sets [#]")
    plot_result["totals"].xaxis.set_major_formatter(FuncFormatter(format_tick_label))
    plot_result["totals"].tick_params(axis="x", which="major", labelsize=params_size)
    plot_result["totals"].xaxis.label.set_size(label_size)
    # Change color to grey
    for child in plot_result["totals"].get_children()[: len(tags)]:
        child.set_color("grey")
    plot_result["totals"].get_children()[0].set_color("dimgray")
    plot_result["totals"].get_children()[1].set_color("dimgray")
    plot_result["totals"].get_children()[2].set_color("dimgray")
    plot_result["totals"].get_children()[3].set_color("dimgray")
    
    # Increase size of the labels in "matrix" (middle plot)
    plot_result["matrix"].tick_params(axis="y", which="major", labelsize=params_size)

    pbar.update(1)
    pbar.set_description('Creating UpSetPlot legend')
    patches = []
    tmp_colors = colors[1:] + [colors[0]]
    for i,tag in enumerate(tags+['Overlap']):
        patches.append(mpatches.Patch(color=tmp_colors[i], label=tag))

    for ax in plt.gcf().axes:
        for text in ax.texts:
            if '%' in text.get_text():  # Identify percentage texts
                text.set_bbox(dict(facecolor='white', pad=-2,edgecolor='white',alpha=0.75))
    
    plt.legend(handles=patches, bbox_to_anchor=bbox,framealpha=1)

    plt.title(title,fontsize=12)
    # save and show
    pbar.update(1)
    pbar.set_description('Saving UpSetPlot')
    fig = plt.gcf()
    
    save_plot(fig,fig_name,directory=FIGURES_DIR,autoclose=True)

    pbar.update(1)
    pbar.close()