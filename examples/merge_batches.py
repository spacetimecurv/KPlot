#########################################################################
# File: merge_batches.py                                                #
# Description: Merge output files from batchtools output folders.       #
#########################################################################

# Import the batch merger class.
from kplot import BatchMerger

# Create a BatchMerger instance.
merger = BatchMerger()

merger.merge(
  file_glob="bhns.co_0.txt",
  output_file="bhns.co_0.txt",
  output_dir="output",
  input_dir="data/batch"
)

merger.merge(
    file_glob="waveforms/rpsi4_*_*.txt",
    group_regex=r"rpsi4_(real|imag)_(\d+)\.txt",
    output_file=lambda k: f"rpsi4_{k[0]}_{k[1]}_master.txt",
    output_dir="output/waveforms",
    input_dir="data/batch"
)
