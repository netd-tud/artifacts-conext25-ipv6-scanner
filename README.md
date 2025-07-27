# artifacts-conext25-ipv6-scanner

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Artifacts for CoNEXT'25 paper #162 A Detailed Measurement View on IPv6 Scanners and Their Adaption to BGP Signals

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ipv6_scanner and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ipv6_scanner   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ipv6_scanner a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

Artifacts: A Detailed Measurement View on IPv6 Scanners and Their Adaption to BGP Signals
===

This repository contains the artifacts for the following paper:
```
ReACKed QUICer: Measuring the Performance of Instant Acknowledgments in QUIC Handshakes
Jonas Mücke, Marcin Nawrocki, Raphael Hiesgen, Thomas C. Schmidt, Matthias Wählisch
Proc. of ACM Internet Measurement Conference (IMC), Madrid, Spain: ACM, 2024
https://doi.org/10.1145/3646547.3689022
```

# Structure
We include all software, data, and analysis scripts required to reproduce our results. 
```
├── 01-quic-go-instant-ack/               <- Fork of `quic-go/quic-go` at e2622bfad865bf4633fb752187c9663402515c6f that implements instant ACK.
├── 02-quic-interop-runner-instant-ack/   <- Fork of `quic-interop/quic-interop-runner` (QIR) at ca27dcb5272a82d994337ae3d14533c318d81b76 with additional configuration options
├── 03-measurement-infra/                 <- Ansible roles and playbook to configure QIR emulation nodes and vantage points of the paper.
│   └── playbook.yaml   <- run with: ansible-playbook -i inventory.yaml playbook.yaml -e @secrets_file.enc --ask-vault-pass 
├── 04-go-pto-tool/                       <- Go tool to link send and received packets in qlog files.
│   └── main.go         <- build with: CGO_ENABLED=0 go build -ldflags="-extldflags=-static" 
└── 05-instant-ack-ccds/                  <- Cookiecutter Data Science project,
    └── README.md      <- Readme with instructions on how to reproduce the paper graphs. 
```
# Reproduction of paper artifacts

Requirements: 256 GB of memory

Clone this repository, then: 
1. Make sure python and wireshark is installed.
1. `cd 05-instant-ack-ccds`
2. Make a virtual environment: `make python_env`
3. Activate python env: `source .venv/bin/activate`
4. Download required data from [https://doi.org/10.25532/OPARA-615](https://doi.org/10.25532/OPARA-615) and extract tars according to `05-instant-ack-ccds/README.md`

Now you can execute the existing notebooks with: 

5. `make nbconvert-execute`

Or if you want to do all preprocessing steps:

5. `make data`

For details or processing only a subset see `05-instant-ack-ccds/README.md`