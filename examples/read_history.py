#########################################################################
# File: read_history.py                                                 #
# Description: Read history data from a file.                           #
#########################################################################

# Import the History class.
from kplot import History

# Import necessary third-party libraries.
import matplotlib.pyplot as plt
import numpy as np

# Call the History constructor.
history = History("data/history/bhns.mhd.hst")

# Load the history data.
history.load_history_data()
data = history.history_data
print(data.keys())

# Plot the data.
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(data["time"], np.abs(1-data["mass"]/data["mass"][0]))
ax.set_yscale("log")
ax.set_xlim([np.min(data["time"]),np.max(data["time"])])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$|1-M_b(t)/M_b(0)|$")

fig.suptitle("Mass conservation")
fig.tight_layout()
plt.show()