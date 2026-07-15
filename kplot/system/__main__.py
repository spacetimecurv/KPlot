"""Allow `python3 -m kplot.system` to keep driving the full-run visualizer.

This is the fallback used by scripts/system/plot_system.sh when the kplot-system
console script is not on PATH.
"""

from .plotter import main

if __name__ == "__main__":
    main()
