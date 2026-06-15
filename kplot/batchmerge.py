#########################################################################
# File: batchmerge.py                                                   #
# Description: Merges output files from batchtools output folders.      #
#########################################################################

# Import necessary standard libaries.
import re
from pathlib import Path
from collections import defaultdict

# Define a function that splits headers.
def split_header(lines):
  header = [l for l in lines if l.startswith("#")]
  data = [l for l in lines if not l.startswith("#")]

  return header, data

# Define the batch merge class.
class BatchMerger:
  """
  A class to merge output files from batchtools output folders.
  """

  def __init__(self, base_pattern="output-*"):
    """
    Initialize the BatchMerger with the path to the batch folders.

    Parameters:
    base_pattern (str): The pattern to match the batch folders.
    """
    self.base_pattern = base_pattern

  def merge(self, file_glob, output_file, strip_header=True, group_regex=None,
    output_dir=".", input_dir="."):
    """
    Generic merge function.

    Parameters:
    file_glob : File pattern inside each folder, e.g. "<jobname>.co_0.txt" or "waveforms/*.txt".
    output_file : Output filename or function(key)->filename if grouping is used.
    strip_header : Whether to keep only first header.
    group_regex : Optional regex with capture groups for grouping files.
    output_dir : Where to write merged output.
    input_dir : Where to read input files.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    folders = sorted(input_dir.glob(self.base_pattern))

    grouped = defaultdict(list)
    headers = {}

    regex = re.compile(group_regex) if group_regex else None

    for folder in folders:
      for file_path in folder.glob(file_glob):
        if regex:
          m = regex.search(file_path.name)
          if not m:
            continue
          key = m.groups()
        else:
          key = ("__all__",)

        lines = file_path.read_text().splitlines(keepends=True)

        if strip_header:
          header, data = split_header(lines)
        else:
          header, data = [], lines

        if key not in headers:
          headers[key] = header

        grouped[key].extend(data)

    for key, data in grouped.items():
      if callable(output_file):
        name = output_file(key)
      else:
        name = output_file

      out_path = output_dir / name

      content = []
      if strip_header and key in headers:
        content.extend(headers[key])

      content.extend(data)

      out_path.write_text("".join(content))
      print(f"Wrote {out_path}")
