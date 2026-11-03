"""List the checkpoints of a simulation."""

# Import necessary standard libaries.
import argparse
import glob
import os
import struct
import sys

# Header written after the '<par_end>' line by RestartOutput::WriteOutputFile:
# int nmb_total, int root_level, RegionSize mesh_size (9 Reals),
# RegionIndcs mesh_indcs (19 ints), RegionIndcs mb_indcs (19 ints),
# Real time, Real dt, int ncycle.
TIME_OFFSET = 8 + 9 * 8 + 2 * 19 * 4


def read_checkpoint(rst_path: str) -> tuple[float, float, int]:
  """
  Read time, dt and cycle from the header of one .rst file.

  Parameters:
  rst_path (str): Path to the .rst checkpoint file.

  Returns:
  (time, dt, cycle) of the checkpoint.
  """
  with open(rst_path, "rb") as f:
    head = f.read(65536)
    loc = head.find(b"<par_end>")
    if loc < 0:
      raise ValueError("no '<par_end>' marker found")
    f.seek(head.find(b"\n", loc) + 1 + TIME_OFFSET)
    time, dt, cycle = struct.unpack("<ddi", f.read(20))
  return time, dt, cycle


def find_segments(simpath: str, batchtools: bool = True) -> list[str]:
  """
  Find the output-XXXX segment directories of a simulation.

  Parameters:
  simpath (str): Simulation directory.
  batchtools (bool): Whether simpath holds batchtools output-XXXX segments.

  Returns:
  Sorted list of segment directories, or [simpath] if there are none.
  """
  if not batchtools:
    return [simpath]
  dirs = sorted(d for d in glob.glob(os.path.join(simpath, "output-[0-9]*"))
                if os.path.isdir(d))
  return dirs or [simpath]


def list_checkpoints(simpath: str, batchtools: bool = True) -> None:
  """
  Print time, dt and cycle of every checkpoint, grouped by segment.

  Parameters:
  simpath (str): Simulation directory.
  batchtools (bool): Whether simpath holds batchtools output-XXXX segments.

  Returns:
  None.
  """
  simpath = os.path.abspath(os.path.expanduser(simpath))
  print("Simulation: %s" % simpath)

  for segment in find_segments(simpath, batchtools):
    rstdir = os.path.join(segment, "rst")
    if not os.path.isdir(rstdir):
      rstdir = segment
    files = sorted(glob.glob(os.path.join(rstdir, "*.rst")) +
                   glob.glob(os.path.join(rstdir, "rank_*", "*.rst")))

    print("\n%s  (%d checkpoint%s)" % (os.path.basename(segment), len(files),
                                       "" if len(files) == 1 else "s"))
    for path in files:
      try:
        time, dt, cycle = read_checkpoint(path)
      except (OSError, ValueError, struct.error) as exc:
        print("  %-24s  %s" % (os.path.basename(path), exc))
        continue
      print("  %-24s  t = %14.6f  dt = %11.4e  cycle = %9d"
            % (os.path.basename(path), time, dt, cycle))


def main(argv: list[str] | None = None) -> int:
  """
  Command-line entry point of kplot-checkpoints.

  Parameters:
  argv (list[str]): Command-line arguments (default: sys.argv).

  Returns:
  Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(
    prog="kplot-checkpoints",
    description="List the restart checkpoints of each output-XXXX segment of "
                "an AthenaK simulation.")
  parser.add_argument("simpath", nargs="?", default=".",
                      help="Simulation directory (default: current directory)")
  parser.add_argument("--batchtools", action=argparse.BooleanOptionalAction, default=True,
                      help="Read checkpoints from output-XXXX subdirs of simpath "
                           "(default); --no-batchtools reads them from simpath")
  args = parser.parse_args(argv)

  if not os.path.isdir(args.simpath):
    print("ERROR: %s is not a directory" % args.simpath, file=sys.stderr)
    return 1
  list_checkpoints(args.simpath, args.batchtools)
  return 0


if __name__ == "__main__":
  sys.exit(main())
