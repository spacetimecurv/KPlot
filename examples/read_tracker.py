#########################################################################
# File: read_tracker.py                                                 #
# Description: Read tracker data from a file.                           #
#########################################################################

# Import the horizon finder class.
from kplot.system.tracker import Tracker

# Import necessary third-party libraries.
import matplotlib.pyplot as plt
import numpy as np

# Call the Tracker constructor.
tracker = Tracker(
  'data/tracker/bhns.co_0.txt',
  'data/tracker/bhns.co_1.txt')

# Load the tracker data.
tracker.load_tracker_data()
tracker_data = tracker.tracker_data

# Plot the tracker data.
tracker.plot_tracker(
  output_path=None,
  save=False)

# Compute the eccentricity.
_, _, _ = tracker.calc_eccentricity(
  t_fit_min=550.0,
  t_fit_max=1350.0,
  t_min_plot=0.0,
  t_max_plot=1600.0,
  init_guess=[20.0, -0.01, 1.0e-6, 0.02, 0.012, 0.0],
  bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0.008, -2*np.pi], 
          [np.inf, np.inf, np.inf, np.inf, 0.020, 2*np.pi]),
  output_path=None,
  save=False)