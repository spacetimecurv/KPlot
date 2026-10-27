#########################################################################
# File: parfile.py                                                      #
# Description: Some utility for dealing with AthenaK parfiles.          #
#########################################################################

# Built-in libraries.
from pathlib import Path
import re

# Define the Parfile class.
class Parfile:
  """
  Small wrapper class for reading and dealing with AthenaK
  parfiles.
  """

  def __init__(self, parfile_path):
    """
    Initialize Parfile class object, by reading the
    specified parfile.

    Parameters:
    parfile_path (str): Path to a parfile holding information about
                        the simulation parameters.
    """
    self.parfile_path = Path(parfile_path)
    self.read_parfile()

  # Read parfile parameters, if existing.
  def read_parfile(self):
    """
    Read parfile and store the parfile parameters.

    Parameters:
    None

    Returns;
    Dict class member with the parfile blocks and parameter values.
    """
    if not self.parfile_path.is_file():
      raise SystemExit(f"File {self.parfile_path} does not exist!")

    # Read in the contents.
    par, block = {}, None
    with open(self.parfile_path) as f:
      for line in f:
        line = line.split('#', 1)[0].strip()
        if not line:
          continue
        m = re.match(r'<(.+)>', line)
        if m:
          block = m.group(1).strip()
          par.setdefault(block, {})
        elif '=' in line and block is not None:
          k, v = line.split('=', 1)
          par[block][k.strip()] = v.strip()

    self.parfile_dict = par
    if par:
      print("$ Parfile loaded successfully...")

  # Small getter function to get parameter value.
  def get(self, block, key, default=None, typ=float):
    """
    Getter function to return the value of a parameter within
    a certain block of the parfile.

    Parameters:
    block   (str): Parfile <block>.
    key     (str): Parameter name.
    default (str): Default value if value cannot be fetched.
    typ    (type): Type of the returned parameter.

    Returns:
    Returns parameter value of type 'typ'.
    """
    try:
      return typ(self.parfile_dict[block][key])
    except KeyError:
      if default is None:
        raise SystemExit(f"Missing <{block}>/{key} in parfile!")
      return default