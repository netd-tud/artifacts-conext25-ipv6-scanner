

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

Requirements: 32 GB of RAM, 32 GB free disk space

Clone this repository, then: 
1. Make sure python 3.10 is installed.
2. Make a virtual environment: `make python_env`
3. Activate python env: `source .venv/bin/activate`
4. Download required data from [https://doi.org/10.5281/zenodo.16419096](https://doi.org/10.5281/zenodo.16419096)
5. Move the `telescope-t*.parquet` files into `./data/processed/`
5. Move the `addr_type*.gz` files into `./data/processed/`
5. Move the `telescope` parquet files into `./data/processed/`
6. To get a clean starting environment run `make clean` first.

Now you can reproduce the paper plots with: 

7. `make plots`

The plots are then stored under `reports/figures/`

To reproduce the paper tables you can simply run:

8. `make nbconvert-clean-execute`
