"""Command-line arguments shared by the analysis scripts."""

# Built-in libraries.
import argparse


def bool_arg(value: str) -> bool:
  """
  Parse a boolean flag value; '0' and 'False' from config.ini mean False.

  Parameters:
  value (str): value to be parsed.

  Returns:
  True, False or error depending on the value.
  """
  v = value.strip().lower()
  if v in ('1', 'true', 'yes', 'on'):
    return True
  if v in ('0', 'false', 'no', 'off', ''):
    return False
  raise argparse.ArgumentTypeError(f"Expected a bool, got '{value}'!")


def sphere_parser(doc: str, radius: float = None, radii: list = None, plot: bool = False) \
                  -> argparse.ArgumentParser:
  """
  Parser with the options every sph analysis uses.  Give `radius` for a single
  --radius, or `radii` for multiple ones; `plot` adds --plot, --plot-only and
  --t-merger and makes --sph-dir optional (check with `require_sph`).

  Parameters:
  doc (str): doc-string explaining the analysis.
  radius (float): single radius of the analysis.
  radii   (list): list of radii.
  plot    (bool): whether to plot or not.

  Returns:
  argparse.ArgumentParser object.
  """
  p = argparse.ArgumentParser(description=doc,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--sph-dir", action="append", default=[], required=not plot,
                 dest="sph_dirs", metavar="DIR",
                 help="AthenaK output-XXXX/sph directory (repeat for each segment)"
                      + ("; required unless --plot-only." if plot else "."))
  p.add_argument("--output-dir", required=True,
                 help="Directory for the outputs and figures.")
  if radii is not None:
    p.add_argument("--radius", type=float, action="append", default=[], dest="radii",
                   help="SPH extraction radius [M_sun] (repeat for several). "
                        f"Default: {', '.join(f'{r:g}' for r in radii)}.")
  else:
    p.add_argument("--radius", type=float, default=radius,
                   help="SPH extraction radius [M_sun]; picks the surface out of "
                        f"dumps holding several radii. Default: {radius:g}.")
  p.add_argument("--jobname", default="bns",
                 help="AthenaK job name prefixing the VTK files. Default: bns.")
  p.add_argument("--n-workers", type=int, default=None,
                 help="Worker processes for the snapshot loop. Default: min(8, cpu_count()).")
  if plot:
    p.add_argument("--plot", action="store_true", help="Plot after the analysis.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip the analysis and plot the existing outputs in --output-dir.")
    p.add_argument("--t-merger", type=float, default=None,
                   help="Merger time [M_sun]; the plot then uses t - t_merger. "
                        "Default: absolute time.")
  return p


def require_sph(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
  """
  Check whether sph-dirs are required.

  Parameters:
  p (argparse.ArgumentParser): argparser object.
  args   (argparse.Namespace): parsed arguments.

  Returns:
  Error if sph-dir is required but not given in the ArgumentParser.
  """
  if not args.plot_only and not args.sph_dirs:
    p.error("--sph-dir is required unless --plot-only is given!")
