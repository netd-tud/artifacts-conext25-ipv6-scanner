#!/bin/bash
# create .csv.gz dataframe file
./process-flows/telescope-data-processing-lightweight.py $1
# more postprocessing
./process-flows/post_processing_01.py $1
# determine scanner type
./process-flows/classify_scanner.py $1

