#########################################################################
# File: batchutils.py                                                   #
# Description: Merges batchtools segments and removes overlaps.         #
#########################################################################

# Import necessary standard libaries.
import re
from pathlib import Path
from collections import defaultdict
from itertools import chain

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

  def __init__(self, base_pattern="output-*", batchtools=True):
    """
    Initialize the BatchMerger with the path to the batch folders.

    Parameters:
    base_pattern (str): The pattern to match the batch folders.
    """
    self.base_pattern = base_pattern
    self.batchtools = batchtools


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

    folders = sorted(input_dir.glob(self.base_pattern)) if self.batchtools else [input_dir]

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
      out_path.parent.mkdir(exist_ok=True, parents=True)

      content = []
      if strip_header and key in headers:
        content.extend(headers[key])

      content.extend(data)

      out_path.write_text("".join(content))
      print(f"Wrote {out_path}")


  def merge_auto(self, history=True, horizon=True, tracker=True, waveforms=True,
                 strip_header=True, output_dir=".", input_dir="."):
    """
    Automatic merger function that detects history files etc. and
    uses the 'merger' function to directly merge them.

    Parameters:
    history       (bool): Whether to merge .hst files.
    horizon       (bool): Whether to merge .horizon_summary_* files.
    tracker       (bool): Whether to merge .co_*.txt tracker files.
    waveforms     (bool): Whether to merge waveform files.
    strip_header  (bool): Whether to keep only first header.
    output_dir     (str): Where to write merged output.
    input_dir      (str): Where to read input files.
    """

    input_dir = Path(input_dir)
    folders = sorted(input_dir.glob(self.base_pattern)) if self.batchtools else [input_dir]

    # Patterns (CHANGE these if the code changes convention).
    history_pattern  = "*.hst"
    horizon_pattern  = "*.horizon_summary*"
    tracker_pattern  = "*.co_*.txt"
    waveform_pattern = "waveforms/rpsi4_*_*.txt"
    patterns = [p for p, use in zip((history_pattern, horizon_pattern, tracker_pattern, waveform_pattern),
                                    (history, horizon, tracker, waveforms)) if use]

    # Collect the unique file paths (relative to each segment) over all segments.
    names = sorted({f.relative_to(folder).as_posix() for folder in folders
                                                     for p in patterns
                                                     for f in folder.glob(p) if f.is_file()})

    # Merge each file across all segments.
    for name in names:
      self.merge(file_glob=name, output_file=name, strip_header=strip_header,
                 output_dir=output_dir, input_dir=input_dir)


class BatchOverlap:
  """
  A class to remove overlapping data from crashed segments.

  If a segment crashes and the next one restarts from an older checkpoint,
  the data the crashed segment wrote after that checkpoint is stale. The rule
  is that the later segment wins: in segment k everything at or after the
  first entry of any later segment is removed. Snapshots (bin/, sph/) are
  compared by their index per output stem, text files (*.hst, *.txt) by their
  time column. Nothing is deleted, stale data is moved to
  <simpath>/<overlap_dir>/<segment>/.
  """

  # Data lines ending in "<stem>.<index>.<ext>" for the snapshot folders.
  snapshot_regex = re.compile(r"^(?P<stem>.+)\.(?P<index>\d+)\.(?P<ext>bin|vtk)$")
  # Header patterns of the time column, "[1]=time" (hst) or "2:time" (txt).
  time_regex = re.compile(r"\[?(\d+)\]?[=:]time\b")

  def __init__(self, simpath=".", base_pattern="output-*", overlap_dir="overlap",
               snapshot_dirs=("bin", "sph"), text_patterns=("*.hst", "*.txt"),
               tol=1e-8):
    """
    Initialize the BatchOverlap with the path to the simulation.

    Parameters:
    simpath         (str): Folder containing the batch segments.
    base_pattern    (str): The pattern to match the batch folders.
    overlap_dir     (str): Folder (relative to simpath) receiving stale data.
    snapshot_dirs (tuple): Snapshot subfolders compared by index.
    text_patterns (tuple): Text files compared by time.
    tol           (float): Tolerance of the time comparison.
    """
    self.simpath = Path(simpath)
    self.overlap_dir = self.simpath / overlap_dir
    self.snapshot_dirs = snapshot_dirs
    self.text_patterns = text_patterns
    self.tol = tol
    self.segments = sorted(d for d in self.simpath.glob(base_pattern) if d.is_dir())
    if not self.segments:
      raise FileNotFoundError(f"No segment matching {base_pattern} in {simpath}!")


  def stale_snapshots(self):
    """
    Returns {segment: [stale snapshot paths]}, comparing the indices of
    every output stem against the first index of all later segments.
    """
    stale = defaultdict(list)
    for sub in self.snapshot_dirs:
      cutoff = {}
      for segment in reversed(self.segments):
        first = {}
        for f in sorted((segment / sub).glob("*")):
          m = self.snapshot_regex.match(f.name)
          if not m:
            continue
          stem, index = m["stem"], int(m["index"])
          if index >= cutoff.get(stem, float("inf")):
            stale[segment].append(f)
          first[stem] = min(first.get(stem, index), index)
        for stem, index in first.items():
          cutoff[stem] = min(cutoff.get(stem, index), index)

    return stale


  def time_column(self, header):
    """
    Returns the time column given in the header, or None.
    """
    for line in header:
      m = self.time_regex.search(line)
      if m:
        return int(m.group(1)) - 1

    return None


  def stale_lines(self):
    """
    Returns {(segment, name): (kept lines, stale lines)} for every text file
    that contains data at or after the first time of a later segment.
    """
    names = sorted({f.name for segment in self.segments
                           for p in self.text_patterns
                           for f in segment.glob(p) if f.is_file()})
    stale = {}
    for name in names:
      cutoff = float("inf")
      for segment in reversed(self.segments):
        path = segment / name
        if not path.is_file():
          continue
        lines = path.read_text().splitlines(keepends=True)
        col = self.time_column(split_header(lines)[0])
        if col is None:
          continue

        kept, removed, first = [], [], None
        for line in lines:
          try:
            t = float(line.split()[col])
          except (IndexError, ValueError):
            kept.append(line)
            continue
          first = t if first is None else min(first, t)
          (removed if t >= cutoff - self.tol else kept).append(line)

        if removed:
          stale[(segment, name)] = (kept, removed)
        if first is not None:
          cutoff = min(cutoff, first)

    return stale


  def report(self):
    """
    Prints the stale data of every segment without changing anything.
    """
    snapshots = self.stale_snapshots()
    lines = self.stale_lines()
    if not snapshots and not lines:
      print("No overlap found.")
      return

    for segment in self.segments:
      for sub in self.snapshot_dirs:
        files = [f for f in snapshots.get(segment, []) if f.parent.name == sub]
        if files:
          stems = sorted({self.snapshot_regex.match(f.name)["stem"] for f in files})
          print(f"{segment.name}/{sub}: {len(files)} stale files in {len(stems)} outputs")
          for stem in stems:
            idx = sorted(int(self.snapshot_regex.match(f.name)["index"]) for f in files
                         if self.snapshot_regex.match(f.name)["stem"] == stem)
            print(f"  {stem}: {idx[0]:05d}-{idx[-1]:05d}")
      for (seg, name), (kept, removed) in lines.items():
        if seg == segment:
          print(f"{segment.name}/{name}: {len(removed)} stale of {len(kept) + len(removed)} lines")


  def clean(self, dry_run=True):
    """
    Moves stale snapshots and stale text lines to <overlap_dir>/<segment>/.

    Parameters:
    dry_run (bool): Only report what would be done.
    """
    self.report()
    if dry_run:
      print("Dry run, nothing changed. Use clean(dry_run=False) to apply.")
      return

    for segment, files in self.stale_snapshots().items():
      for f in files:
        target = self.overlap_dir / segment.name / f.parent.name / f.name
        target.parent.mkdir(exist_ok=True, parents=True)
        f.rename(target)

    for (segment, name), (kept, removed) in self.stale_lines().items():
      target = self.overlap_dir / segment.name / name
      target.parent.mkdir(exist_ok=True, parents=True)
      with open(target, "a") as f:
        f.write("".join(removed))
      (segment / name).write_text("".join(kept))

    print(f"Moved stale data to {self.overlap_dir}")
