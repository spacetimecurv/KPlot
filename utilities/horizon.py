#########################################################################
# File: horizon.py                                                      #
# Description: Horizon utilities for the AthenaK horizon finder.        #
#########################################################################

# Import necessary standard libaries.
from fileinput import filename
import re
import os

# Import necessary third-party libraries.
import numpy as np

# Define the horizon finder class.
class HorizonFinder:
    """
    A class to handle output from the AthenaK horizon finder.
    """

    def __init__(self, horizon_path):
        """
        Initialize the HorizonFinder with the path to the horizon file.

        Parameters:
        horizon_path (str): The path to the horizon file.
        """
        self.horizon_path = horizon_path
        self.horizon_data = None

    def load_horizon_data(self):
        """
        Load the horizon data from the specified file path.

        Returns:
        Dictionary holding the horizon data.
        """
        if not os.path.exists(self.horizon_path):
            raise FileNotFoundError(f"Horizon file not found at: {self.horizon_path}")

        # Find the headers.
        with open(self.horizon_path) as f:
            header_line = f.readline().strip()

        headers = re.findall(r'\d+:([^\s]+)', header_line)

        # Read the data with numpy.
        data = np.loadtxt(self.horizon_path, comments='#')

        # Create a dictionary.
        data_dict = {name: data[:, i] for i, name in enumerate(headers)}
        self.horizon_data = data_dict








