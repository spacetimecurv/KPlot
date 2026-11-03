"""Small plotting helpers shared by the analysis scripts."""

# Built-in libraries.
from typing import Callable

# KPlot utility.
from .constants import MSUN_TO_MS


def time_axis(t_merger: float = None) -> tuple[Callable, str]:
  """
  (t [M_sun] -> plot time [ms], x label), relative to t_merger if given.

  Parameters:
  t_merger (float): merger time in code units.

  Returns:
  tuple with a callable function for shifting the time array to
  the respective fix point and the y-axis label.
  """
  t0 = 0.0 if t_merger is None else t_merger
  label = r'$t$ [ms]' if t_merger is None else r'$t - t_\mathrm{merger}$ [ms]'
  return (lambda t: (t - t0) * MSUN_TO_MS), label


def savefig(fig, path: str, tight: bool = True) -> None:
  """
  Wrapper function for saving matplotlib figures.

  Parameters:
  fig (plt.figure): matplotlib figure object.
  path       (str): path where figure should be stored.
  tight     (bool): whether to have a tight layout.

  Returns: None
  """

  import matplotlib.pyplot as plt
  if tight:
    fig.tight_layout()
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {path}')
