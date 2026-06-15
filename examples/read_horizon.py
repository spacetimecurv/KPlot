#########################################################################
# File: read_horizon.py                                                 #
# Description: Read horizon data from a file.                           #
#########################################################################

# Import the horizon finder class.
from utilities import HorizonFinder

# Import necessary third-party libraries.
import matplotlib.pyplot as plt
import numpy as np

# Call the HorizonFinder constructor.
horizon_finder = HorizonFinder('data/horizon/z4c.horizon_summary_0.txt')

# Load the horizon data.
horizon_finder.load_horizon_data()
horizon_data = horizon_finder.horizon_data

# Plot the horizon data.
fig, axes = plt.subplots(2,2,figsize=(12,6))
axes[0,0].plot(horizon_data["time"], horizon_data["mass"], color="red")
axes[0,0].set_xlim([np.min(horizon_data["time"]), np.max(horizon_data["time"])])
axes[0,0].set_xlabel(r"$t$")
axes[0,0].set_ylabel(r"$M_{\mathrm{horizon}}$")

axes[0,1].plot(horizon_data["time"], horizon_data["Sz"], color="blue")
axes[0,1].set_xlim([np.min(horizon_data["time"]), np.max(horizon_data["time"])])
axes[0,1].set_xlabel(r"$t$")
axes[0,1].set_ylabel(r"$s_{\mathrm{z,horizon}}$")
axes[0,1].set_yscale("log")

axes[1,0].plot(horizon_data["time"], horizon_data["S"], color="green")
axes[1,0].set_xlim([np.min(horizon_data["time"]), np.max(horizon_data["time"])])
axes[1,0].set_xlabel(r"$t$")
axes[1,0].set_ylabel(r"$S_{\mathrm{horizon}}$")
axes[1,0].set_yscale("log")

axes[1,1].plot(horizon_data["time"], horizon_data["area"], color="brown", label="AthenaK")
axes[1,1].set_xlim([np.min(horizon_data["time"]), np.max(horizon_data["time"])])
axes[1,1].set_xlabel(r"$t$")
axes[1,1].set_ylabel(r"$A_{\mathrm{horizon}}$")
axes[1,1].set_yscale("log")

fig.suptitle("Spinning-Puncture test")
fig.tight_layout()
plt.show()
