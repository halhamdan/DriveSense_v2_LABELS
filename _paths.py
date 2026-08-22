"""
Shared local paths for the dataset-construction and technical-validation
pipeline scripts in this repository.

These pipeline stages were run once, by the authors, against a pre-publication
staging directory tree -- they are included for methodological transparency
(showing exactly how the released dataset was built and validated), not as a
one-command reproduction of the already-published data. All paths below are
read from environment variables rather than hardcoded to the authors'
machine; set whichever ones a given script needs (see README.md's
"Environment variables" table) before running it.
"""
import os
from pathlib import Path


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name, ""))


# Author's pre-publication staging tree:
STAGING_LABELED_DIR = _env_path("STAGING_LABELED_DIR")   # output of 01_annotation (D{n}_S{n}_LABELED.csv, incl. raw GPS)
STAGING_FUSED_DIR = _env_path("STAGING_FUSED_DIR")        # output of 02_dataset_construction (D{n}/Session_{n}/D{n}_S{n}_fused.csv, pre-trim)
STAGING_VIDEO_ROOT = _env_path("STAGING_VIDEO_ROOT")       # raw + blurred video before session trimming
STAGING_DATA_ROOT = _env_path("STAGING_DATA_ROOT")         # general staging root used by the trim-screening scripts

# The final, released dataset (what a reader downloads):
PUBLISHED = _env_path("STAGING_FUSED_DIR")   # kept as an alias since several scripts import this name directly
DATASET_ROOT = _env_path("DATASET_ROOT")     # Raw_Dataset/ and Preprocessed_Dataset/ live directly under this
