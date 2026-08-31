#########################################################################
# File: read_waveforms.py                                               #
# Description: Read waveform data from a file.                          #
#########################################################################

# Import the waveform reader class.
from kplot.system.waveforms import Waveform
from kplot.sphere.mergertime import merger_time

# Import necessary third-party libraries.
import matplotlib.pyplot as plt
import numpy as np

# Built-in libraries.
import os

# Call the Waveform constructor.
path = '/home/no96soq/athenak/runs/PhysicsComparisonBHNS/Res_MHD_M1/'
waveform = Waveform(os.path.join(path, 'output-0000/waveforms'), '0400', 6.373609, 0.007871519412915)
waveform.load_psi4_data()

# Compute the kick.
t_merge, _ = merger_time(path, 400.0)
waveform.compute_kick(
  t_merge=t_merge,
  insp_min=300.0,
  insp_max=-200.0,
  cutoff=None,
  hodograph=True,
  output_dir='/home/no96soq/athenak/runs/PhysicsComparisonBHNS/Res_MHD_M1'
)

waveform.strain(mode_key='22')
waveform.plot_strain(0.0, 2000.0)

