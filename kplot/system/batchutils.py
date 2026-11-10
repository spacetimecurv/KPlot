"""Utility for handling batchtools segments."""

# Import necessary standard libaries.
import re
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable
from itertools import chain

# Define a function that splits headers.
def split_header(lines: list[str]) -> tuple[list[str], list[str]]:
  """
  Split lines into '#' header lines and data lines.

  Parameters:
  lines (list[str]): Lines of a text file.

  Returns:
  (header lines, data lines).
  """
  header = [l for l in lines if l.startswith("#")]
  data = [l for l in lines if not l.startswith("#")]

  return header, data

# Define the batch merge class.
class BatchMerger:
  """
  A class to merge output files from batchtools output folders.
  """

  def __init__(self, base_pattern: str = "output-*", batchtools: bool = True) -> None:
    """
    Initialize the BatchMerger with the path to the batch folders.

    Parameters:
    base_pattern (str): The pattern to match the batch folders.
    batchtools (bool): Whether the data is split into batchtools segments.
    """
    self.base_pattern = base_pattern
    self.batchtools = batchtools


  def merge(self, file_glob: str, output_file: str | Callable[[tuple], str],
            strip_header: bool = True, group_regex: str | None = None,
            output_dir: str = ".", input_dir: str = ".") -> None:
    """
    Generic merge function.

    Parameters:
    file_glob (str): File pattern inside each folder, e.g. "<jobname>.co_0.txt" or "waveforms/*.txt".
    output_file (str): Output filename or function(key) -> filename if grouping is used.
    strip_header (bool): Whether to keep only first header.
    group_regex (str): Optional regex with capture groups for grouping files.
    output_dir (str): Where to write merged output.
    input_dir (str): Where to read input files.

    Returns:
    None.
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


  def merge_auto(self, history: bool = True, horizon: bool = True, tracker: bool = True,
                 waveforms: bool = True, shape: bool = True, output_dir: str = ".",
                 input_dir: str = ".") -> None:
    """
    Automatic merger function that detects history files etc. and
    uses the 'merger' function to directly merge them.

    Parameters:
    history       (bool): Whether to merge .hst files.
    horizon       (bool): Whether to merge .horizon_summary_* files.
    tracker       (bool): Whether to merge .co_*.txt tracker files.
    waveforms     (bool): Whether to merge waveform files.
    shape         (bool): Whether to merge horizon shape files.
    output_dir     (str): Where to write merged output.
    input_dir      (str): Where to read input files.

    Returns:
    None.
    """

    input_dir = Path(input_dir)
    folders = sorted(input_dir.glob(self.base_pattern)) if self.batchtools else [input_dir]

    # Patterns (CHANGE these if the code changes convention).
    # (Pattern, use, strip_header).
    patterns = [("*.hst",                   history,   True),
                ("*.horizon_summary*",      horizon,   True),
                ("*.co_*.txt",              tracker,   True),
                ("waveforms/rpsi4_*_*.txt", waveforms, True),
                ("*.horizon_shape_*.txt",   shape,     False)]

    # Collect the unique file paths (relative to each segment) over all segments.
    names = {f.relative_to(folder).as_posix(): strip  for folder in folders
                                                      for p, use, strip in patterns if use
                                                      for f in folder.glob(p) if f.is_file()}

    # Merge each file across all segments.
    for name, strip in sorted(names.items()):
      self.merge(file_glob=name, output_file=name, strip_header=strip,
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
  # Entry time in comment lines, "# iter = 3, Time = 1600" (horizon shape).
  entry_time_regex = re.compile(r"Time\s*=\s*([-+0-9.eE]+)")

  def __init__(self, simpath: str = ".", base_pattern: str = "output-*",
               overlap_dir: str = "overlap", snapshot_dirs: tuple = ("bin", "sph"),
               text_patterns: tuple = ("*.hst", "*.txt", "waveforms/*.txt"), tol: float = 1e-8) -> None:
    """
    Initialize the BatchOverlap with the path to the simulation.

    Parameters:
    simpath         (str): Folder containing the batch segments.
    base_pattern    (str): The pattern to match the batch folders.
    overlap_dir     (str): Folder (relative to simpath) receiving stale data.
    snapshot_dirs (tuple): Snapshot subfolders compared by index.
    text_patterns (tuple): Text files compared by time (globs relative to each segment).
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


  def stale_snapshots(self) -> dict[Path, list[Path]]:
    """
    Find snapshots that a later segment overwrites, comparing the indices of
    every output stem against the first index of all later segments.

    Returns:
    {segment: [stale snapshot paths]}.
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


  def line_times(self, lines: list[str], col: int | None) -> list[float | None]:
    """
    Time of every line: from the time column, or, without one, from the
    preceding "Time =" comment.

    Parameters:
    lines (list[str]): Lines of a text file.
    col  (int | None): Index of the time column.

    Returns:
    List of times (None for lines without a time).
    """
    times, entry = [], None
    for line in lines:
      if col is None:
        m = self.entry_time_regex.search(line) if line.startswith("#") else None
        entry = float(m.group(1)) if m else entry
        times.append(entry)
        continue
      try:
        times.append(float(line.split()[col]))
      except (IndexError, ValueError):
        times.append(None)

    return times


  def time_column(self, header: list[str]) -> int | None:
    """
    Find the time column given in the header.

    Parameters:
    header (list[str]): Header lines of a text file.

    Returns:
    Index of the time column, or None.
    """
    for line in header:
      m = self.time_regex.search(line)
      if m:
        return int(m.group(1)) - 1

    return None


  def stale_lines(self) -> dict[tuple[Path, str], tuple[list[str], list[str]]]:
    """
    Find the text lines at or after the first time of a later segment.

    Returns:
    {(segment, name): (kept lines, stale lines)} for every affected text file.
    """
    names = sorted({f.relative_to(segment).as_posix() for segment in self.segments
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
        times = self.line_times(lines, col)
        if all(t is None for t in times):
          continue

        kept, removed, first = [], [], None
        for line, t in zip(lines, times):
          if t is None:
            kept.append(line)
            continue
          first = t if first is None else min(first, t)
          (removed if t >= cutoff - self.tol else kept).append(line)

        if removed:
          stale[(segment, name)] = (kept, removed)
        if first is not None:
          cutoff = min(cutoff, first)

    return stale


  def report(self) -> None:
    """
    Prints the stale data of every segment without changing anything.

    Returns:
    None.
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


  def clean(self, dry_run: bool = True) -> None:
    """
    Moves stale snapshots and stale text lines to <overlap_dir>/<segment>/.

    Parameters:
    dry_run (bool): Only report what would be done.

    Returns:
    None.
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
