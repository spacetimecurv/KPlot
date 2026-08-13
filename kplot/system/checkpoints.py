#########################################################################
# File: checkpoints.py                                                  #
# Description: Lists the restart checkpoints of an AthenaK simulation.  #
#########################################################################

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


def read_checkpoint(rst_path):
  """Return (time, dt, cycle) read from the header of one .rst file."""
  with open(rst_path, "rb") as f:
    head = f.read(65536)
    loc = head.find(b"<par_end>")
    if loc < 0:
      raise ValueError("no '<par_end>' marker found")
    f.seek(head.find(b"\n", loc) + 1 + TIME_OFFSET)
    time, dt, cycle = struct.unpack("<ddi", f.read(20))
  return time, dt, cycle


def find_segments(simpath):
  """Return the sorted output-XXXX dirs under simpath, or simpath itself."""
  dirs = sorted(d for d in glob.glob(os.path.join(simpath, "output-[0-9]*"))
                if os.path.isdir(d))
  return dirs or [simpath]


def list_checkpoints(simpath):
  """Print time, dt and cycle of every checkpoint, grouped by segment."""
  simpath = os.path.abspath(os.path.expanduser(simpath))
  print("Simulation: %s" % simpath)

  for segment in find_segments(simpath):
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


def main(argv=None):
  parser = argparse.ArgumentParser(
    prog="kplot-checkpoints",
    description="List the restart checkpoints of each output-XXXX segment of "
                "an AthenaK simulation.")
  parser.add_argument("simpath", nargs="?", default=".",
                      help="Simulation directory (default: current directory)")
  args = parser.parse_args(argv)

  if not os.path.isdir(args.simpath):
    print("ERROR: %s is not a directory" % args.simpath, file=sys.stderr)
    return 1
  list_checkpoints(args.simpath)
  return 0


if __name__ == "__main__":
  sys.exit(main())
