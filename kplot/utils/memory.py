"""Peak memory diagnostics of the current process."""


def reset_peak_rss() -> None:
  """Reset the peak resident memory (VmHWM) of this process (Linux >= 4.0)."""
  try:
    with open("/proc/self/clear_refs", "w") as fp:
      fp.write("5")
  except OSError:
    pass


def peak_rss_gb() -> float:
  """Peak resident memory (VmHWM) of this process since the last reset [GB]."""
  with open("/proc/self/status") as fp:
    for line in fp:
      if line.startswith("VmHWM:"):
        return int(line.split()[1]) / 1024**2
  return float("nan")
