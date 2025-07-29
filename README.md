

Artifacts: A Detailed Measurement View on IPv6 Scanners and Their Adaption to BGP Signals
===

This repository contains the artifacts for the following paper:
```
A Detailed Measurement View on IPv6 Scanners and Their Adaption to BGP Signals
Isabell Egloff, Raphael Hiesgen, Maynard Koch, Thomas C. Schmidt, and Matthias Wählisch
Proc. ACM Netw., Vol. 3, No. CoNEXT3, Article 15. Publication date: September 2025.
https://doi.org/10.1145/3749215
```

# Reproduction of paper artifacts

Requirements: 64 GB of RAM, 50 GB free disk space

Clone this repository, then: 
1. Make sure python 3.10 is installed.
2. Make a virtual environment: `make python_env`
3. Activate python env: `source .venv/bin/activate`
4. Download required data from [https://doi.org/10.5281/zenodo.16419096](https://doi.org/10.5281/zenodo.16419096)
5. Move the `telescope-t*.parquet` files into `./data/processed/`
5. Move the `addr_type*.gz` files into `./data/processed/`
6. To get a clean starting environment run `make clean` first.

Now you can reproduce the paper plots with: 

7. `make plots`

The plots are then stored under `reports/figures/`

To reproduce the paper tables you can simply run:

8. `make nbconvert-clean-execute`

## Cleaning the environment
- `make clean` to remove figures and the table.html file.
- `make clean-cache` to remove the cached pickle files for faster plot rendering
- It is necessary to run these commands when creating new dataframes from the raw files (see below)

## Creating dataframes from raw data
- You can recreate parts of the dataframes with the following commands
- `make t1-from-raw` to create the telescope T1 dataframe
- `make t2-from-raw` to create the telescope T2 dataframe
- `make t3-from-raw` to create the telescope T3 dataframe
- `make t4-from-raw` to create the telescope T4 dataframe
- Clean environment (see above)
- `make plots-new`
- `make nbconvert-clean-execute-new`

## External tools
1. We include the script to perform RDNS lookups: `tools/rdns-script.sh [input addr file] [outputfile]` (input file contains one ipv6 addr per line)
2. We include the addr6 tool with small changes from the IPv6Toolkit in `tools/ipv6toolkit.tar.gz`
