#########################################################################
# File: bin_convert.py                                                  #
# Description: Binary file reader of athenak/vis/python.                #
#########################################################################

# Third-party libraries.
import numpy as np


def read_binary(filename):
  """
  Reads a bin file from filename to dictionary.

  Originally written by Lev Arzamasskiy (leva@ias.edu) on 11/15/2021
  Updated to support mesh refinement by George Wong (gnwong@ias.edu) on 01/27/2022
  Made faster by Drummond Fielding on 09/09/2024

  args:
    filename - string
        filename of bin file to read

  returns:
    filedata - dict
        dictionary of fluid file data
  """

  filedata = {}

  # load file and get size
  fp = open(filename, "rb")
  fp.seek(0, 2)
  filesize = fp.tell()
  fp.seek(0, 0)

  # load header information and validate file format
  code_header = fp.readline().split()
  if len(code_header) < 1:
    raise TypeError("unknown file format")
  if code_header[0] != b"Athena":
    raise TypeError(
        f"bad file format \"{code_header[0].decode('utf-8')}\" "
        + '(should be "Athena")'
    )
  version = code_header[-1].split(b"=")[-1]
  if version != b"1.1":
    raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

  pheader_count = int(fp.readline().split(b"=")[-1])
  pheader = {}
  for _ in range(pheader_count - 1):
    key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
    pheader[key] = val
  time = float(pheader["time"])
  cycle = int(pheader["cycle"])
  locsizebytes = int(pheader["size of location"])
  varsizebytes = int(pheader["size of variable"])

  nvars = int(fp.readline().split(b"=")[-1])
  var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
  header_size = int(fp.readline().split(b"=")[-1])
  header = [
      line.decode("utf-8").split("#")[0].strip()
      for line in fp.read(header_size).split(b"\n")
  ]
  header = [line for line in header if len(line) > 0]

  if locsizebytes not in [4, 8]:
    raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
  if varsizebytes not in [4, 8]:
    raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

  locfmt = "d" if locsizebytes == 8 else "f"
  varfmt = "d" if varsizebytes == 8 else "f"

  # load grid information from header and validate
  def get_from_header(header, blockname, keyname):
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith("<"):
      blockname = "<" + blockname
    if blockname[-1] != ">":
      blockname += ">"
    block = "<none>"
    for line in [entry for entry in header]:
      if line.startswith("<"):
        block = line
        continue
      key, value = line.split("=")
      if block == blockname and key.strip() == keyname:
        return value
    raise KeyError(f"no parameter called {blockname}/{keyname}")

  Nx1 = int(get_from_header(header, "<mesh>", "nx1"))
  Nx2 = int(get_from_header(header, "<mesh>", "nx2"))
  Nx3 = int(get_from_header(header, "<mesh>", "nx3"))
  nx1 = int(get_from_header(header, "<meshblock>", "nx1"))
  nx2 = int(get_from_header(header, "<meshblock>", "nx2"))
  nx3 = int(get_from_header(header, "<meshblock>", "nx3"))

  nghost = int(get_from_header(header, "<mesh>", "nghost"))

  x1min = float(get_from_header(header, "<mesh>", "x1min"))
  x1max = float(get_from_header(header, "<mesh>", "x1max"))
  x2min = float(get_from_header(header, "<mesh>", "x2min"))
  x2max = float(get_from_header(header, "<mesh>", "x2max"))
  x3min = float(get_from_header(header, "<mesh>", "x3min"))
  x3max = float(get_from_header(header, "<mesh>", "x3max"))

  # load data from each meshblock
  n_vars = len(var_list)
  mb_count = 0

  mb_index = []
  mb_logical = []
  mb_geometry = []

  mb_data = {}
  for var in var_list:
    mb_data[var] = []
  while fp.tell() < filesize:
    mb_index.append(
        np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost
    )
    nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0]) + 1
    nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2]) + 1
    nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4]) + 1
    mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
    mb_geometry.append(
        np.frombuffer(
            fp.read(6 * locsizebytes),
            dtype=np.float64 if locfmt == "d" else np.float32,
        )
    )

    data = np.fromfile(
        fp,
        dtype=np.float64 if varfmt == "d" else np.float32,
        count=nx1_out * nx2_out * nx3_out * n_vars,
    )
    data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
    for vari, var in enumerate(var_list):
      mb_data[var].append(data[vari])
    mb_count += 1

  fp.close()

  filedata["header"] = header
  filedata["time"] = time
  filedata["cycle"] = cycle
  filedata["var_names"] = var_list

  filedata["Nx1"] = Nx1
  filedata["Nx2"] = Nx2
  filedata["Nx3"] = Nx3
  filedata["nvars"] = nvars

  filedata["x1min"] = x1min
  filedata["x1max"] = x1max
  filedata["x2min"] = x2min
  filedata["x2max"] = x2max
  filedata["x3min"] = x3min
  filedata["x3max"] = x3max

  filedata["n_mbs"] = mb_count
  filedata["nx1_mb"] = nx1
  filedata["nx2_mb"] = nx2
  filedata["nx3_mb"] = nx3
  filedata["nx1_out_mb"] = (mb_index[0][1] - mb_index[0][0]) + 1
  filedata["nx2_out_mb"] = (mb_index[0][3] - mb_index[0][2]) + 1
  filedata["nx3_out_mb"] = (mb_index[0][5] - mb_index[0][4]) + 1

  filedata["mb_index"] = np.array(mb_index)
  filedata["mb_logical"] = np.array(mb_logical)
  filedata["mb_geometry"] = np.array(mb_geometry)
  filedata["mb_data"] = mb_data

  return filedata