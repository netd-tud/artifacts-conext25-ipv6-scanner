from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

T1_DATAFRAME = PROCESSED_DATA_DIR / "telescope-t1-20240702.parquet"
T2_DATAFRAME = PROCESSED_DATA_DIR / "telescope-t2-20240702.parquet"
T3_DATAFRAME = PROCESSED_DATA_DIR / "telescope-t3-20240702.parquet"
T4_DATAFRAME = PROCESSED_DATA_DIR / "telescope-t4-20240702.parquet"
TAXONOMY_BEFORE_SPLIT = PROCESSED_DATA_DIR / "taxonomy-before-split-df.csv"
TAXONOMY_DURING_SPLIT = PROCESSED_DATA_DIR / "taxonomy-during-split-t1-df.csv"
ANNOUNCEMENT_LOG_FILE = PROCESSED_DATA_DIR / "prefix-announcements.csv"

NIST_ONEOFF_SUBNET = PROCESSED_DATA_DIR / 'NIST_oneoff_subnet_df.parquet'
NIST_PERIODIC_SUBNET = PROCESSED_DATA_DIR / 'NIST_periodic_subnet_df.parquet'
NIST_INTERMITTENT_SUBNET = PROCESSED_DATA_DIR / 'NIST_intermittent_subnet_df.parquet'
NIST_ONEOFF_IID = PROCESSED_DATA_DIR / 'NIST_oneoff_iid_df.parquet'
NIST_PERIODIC_IID = PROCESSED_DATA_DIR / 'NIST_periodic_iid_df.parquet'
NIST_INTERMITTENT_IID = PROCESSED_DATA_DIR / 'NIST_intermittent_iid_df.parquet'

FIG_NEW_PREFIXES = 'new_prefixes_s2'
FIG_AGG_TELESCOPES_FULL = 'aggregated_full_period_all_telescopes'
FIG_HEAVY_HITTER_BUBBLES = 'heavy-hitter-per-day-bubbles'
FIG_NETWORK_TRAFFIC_OVERVIEW = 'overview-prefixes-squared'
FIG_TAX_BEFORE_SPLIT = 'Taxonomy_before_split'
FIG_TAX_DURING_SPLIT = 'Taxonomy_T1_during_split'
FIG_SESSIONS_PRESPLIT = 'sessions-telescopes-12week'
FIG_SESSIONS_PER_PREFIX = 'scan_sessions_per_prefix'
FIG_SRC_SES_T1_VS_OTHER = 'sources_sessions_all_telescopes_2weeks'
FIG_SUBNET_COVERAGE = 'Subent_Coverage'
FIG_NIST = 'NIST'
FIG_OVERLAP_CUM = 'overlap-cumulative-first-observation'
FIG_OVERLAP_ADDR_ALL = 'overlap-addrs-all-packets-by-telescope'

FIGSIZE_SMALL = (8 * 0.7, 3 * 0.5)
FIGSIZE_SMALL_2 = (8 * 0.7, 5 * 0.7)
FIGSIZE_SMALL_WITH_LEGEND_ON_TOP = (8 * 0.7, 3 * 0.6)
FIGSIZE_MEDIUM = (8 * 0.7, 5.5 * 0.5)
FIGSIZE_WIDE = (14 * 0.7, 5 * 0.5)
FIGSIZE_LONG = (7 * 0.8, 10 * 0.5)
FIGSIZE_SPECIAL = (8 * 0.7, 4 * 0.65)
FIGSIZE_EXTRA_WIDE = (21 * 0.7, 4 * 0.8)
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
