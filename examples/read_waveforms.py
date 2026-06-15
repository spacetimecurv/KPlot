#########################################################################
# File: read_waveforms.py                                               #
# Description: Read waveform data from a file.                          #
#########################################################################

# Import the waveform reader class.
from utilities import Waveform

# Import necessary third-party libraries.
import matplotlib.pyplot as plt
import numpy as np

# Call the Waveform constructor.
waveform = Waveform('data/batch/output-0001/waveforms')
waveform.load_waveform_data()
data = waveform.waveform_data

# Convert to retarded time.
waveform.retarded_time(300.0, 2.8)

# Plot the waveform data.
radius = "0300"
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(data[radius]["time"], data[radius]["real"]["22"], color="red", linestyle="dashed", label=r"$\Re{(\Psi_4)}$")
ax.plot(data[radius]["time"], data[radius]["imag"]["22"], color="blue", linestyle="dashed", label=r"$\Im{(\Psi_4)}$")
ax.set_xlim([np.min(data[radius]["time"]), np.max(data[radius]["time"])])
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$\Psi_4$")
ax.legend()

fig.tight_layout()
plt.show()