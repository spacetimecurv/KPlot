#########################################################################
# File: meshblocks.py                                                   #
# Description: Some MeshBlock utility.                                  #
#########################################################################

# Built-in libraries.
import itertools
from pathlib import Path

# Third-party libraries.
import numpy as np

# KPlot utility.
from kplot.system.parfile import Parfile
from kplot.system.tracker import Tracker, RefineTracker

# Define the MeshBlock class.
class MeshBlock:
  """
  Define MeshBlock class with various utilities targeted at AthenaK simulations.
  """

  def __init__(self, parfile_path = None):
    """
    MeshBlock class initializer.

    Parameters:
    parfile_path (str): Path to a parfile holding information about
                        the simulation mesh
    """
    if parfile_path is not None:
      self.parfile = Parfile(parfile_path)

  # Refinement function that predicts number of MeshBlocks based on
  # refinement criterion (wrapper).
  def predict_meshblocks(self, mode = "disk", **refargs):
    """
    Wrapper function predicting the number of MeshBlocks upon refinement,
    based on the refinement criterion used, and the underlying mesh from the parfile.

    Parameters:
    mode (str): Which refinement criterion is used.
    **refargs : Dictionary of refinement arguments for each refinement function.
    """
    # Add more refinement criteria here.
    choices = ["disk"]
    if mode == "disk":
      # Refinement that covers a disk with the same
      # resolution within a radius R and height z.
      self.__predict_meshblocks_disk(**refargs)
    else:
      raise SystemExit(f"$ Choices are {choices}...")

  # Disk refinement function.
  def __predict_meshblocks_disk(self, **kwargs):
    """
    Function that predicts the number of MeshBlocks upon refinement
    of a criterion, where the same resolution is targeted within
    a radius R and height z, based on the underlying mesh from
    the parfile. Only works with tracker refinement criterion.

    Parameters:
    **kwargs: Needs to contain 'disk-reflevel' (refinement level the disk should
              have), 'disk-rmax' (the radius of the refined region), 'disk-zmax'
              (height of the refined region at 'disk-reflevel'), and
              'disk-tracker' (which tracker to use for the refinement). Path to the
              current tracker files prior to scheduled refinement should be given with
              'disk-tracker-path'. Also needs 'disk-t', i.e. the time at which to refine.
              Optionally, specify the number of nodes via 'nodes' and number of GPUs
              per node 'gpus-per-node' to get the load balancing after refinement.
    """
    print("\nPredicting MeshBlocks after disk refinement R < rmax & |z| < zmax...")
    # Arguments needed for the prediction.
    pf = self.parfile
    args = ["disk-time", "disk-reflevel", "disk-rmax", "disk-zmax",
            "disk-tracker", "disk-tracker-path"]
    if not all(key in kwargs for key in args):
      raise SystemExit(f"Predictor needs {args}!")

    # Check if tracker files exist.
    tracker_path = Path(kwargs["disk-tracker-path"])
    if not tracker_path.is_dir():
      raise SystemExit("Path containing the tracker files does not "
                       f"exist under {tracker_path}!")

    tracker_files = sorted(tracker_path.glob("*.co_[01].txt"))
    if not 1 <= len(tracker_files) <= 2:
      raise SystemExit(f"Expected one or two tracker files under {tracker_path}, "
                       f"found {[f.name for f in tracker_files]}!")

    # Create tracker object.
    tracker = Tracker(*tracker_files)

    # Get the grid setup of base level.
    xmin, xmax = pf.get('mesh', 'x1min'), pf.get('mesh', 'x1max')
    nroot = int(pf.get('mesh', 'nx1') // pf.get('meshblock', 'nx1'))
    maxlev = int(pf.get('mesh_refinement', 'num_levels', typ=int)) - 1
    dx_m = (xmax - xmin) / pf.get('mesh', 'nx1') * 1476.625  # M_sun -> m

    # Get the tracker regions.
    regions = []
    for ico in range(pf.get('z4c', 'nco', 0, int)):
      c = tracker.tracker_pos(ico, float(kwargs["disk-time"]))
      regions.append(RefineTracker(c, pf.get('z4c', f'co_{ico}_radius'),
                                      pf.get('z4c', f'co_{ico}_reflevel', typ=int)))
      print(f"$ Tracker {ico}: pos=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
            f"| r = {regions[-1].r} | level = {regions[-1].level}")
    for nr in range(16):
      if f'radius_{nr}_rad' not in pf.parfile_dict.get('z4c_amr', {}):
        break
      regions.append(OriginRadius(pf.get('z4c_amr', f'radius_{nr}_rad'),
                                  pf.get('z4c_amr', f'radius_{nr}_reflevel', -1, int)))

    # MeshBlocks prior to disk refinement.
    leaves_pre = self.build_octree(regions, xmin, xmax, nroot, maxlev)

    # Define the refinement regions based on the desired refined disk.
    dl = kwargs["disk-reflevel"]
    if dl >= 0:
      rmax = kwargs["disk-rmax"] if kwargs["disk-rmax"] is not None else \
             pf.get('z4c_amr', 'disk_rmax')
      zmax = kwargs["disk-zmax"] if kwargs["disk-zmax"] is not None else \
             pf.get('z4c_amr', 'disk_zmax')
      it = kwargs["disk-tracker"] if kwargs["disk-tracker"] is not None else \
           pf.get('z4c_amr', 'disk_tracker', 0, int)
      c = tracker.tracker_pos(it, kwargs["disk-time"]) if it >= 0 else np.zeros(3)
      regions.append(RefineDisk(c, rmax, zmax, dl))
      print(f"$ Disk: level = {dl} ({dx_m / 2**dl:.0f} m) | R < {rmax} | |z| < {zmax} around tracker {it}")

    # Build the octree.
    leaves = self.build_octree(regions, xmin, xmax, nroot, maxlev)
    levs = np.bincount([L for L, *_ in leaves], minlength = maxlev + 1)
    print(f"\n$ Total MeshBlocks (before refinement) = {len(leaves_pre)}")
    print(f"$ Total MeshBlocks (after refinement) = {len(leaves)}")
    print("$ Tree:")
    for L, n in enumerate(levs):
      if n:
        print(f"    level {L:2d} ({dx_m/2**L:7.0f} m): {n}")

    if ("nodes", "gpus-per-node") in kwargs:
      nr = kwargs["nodes"] * kwargs["gpus-per-node"]
      lim = pf.get('mesh_refinement', 'max_nmb_per_rank', 0, int)
      print(f"\n{nr} ranks -> {len(leaves)/nr:.1f} blocks/rank"
            + (f" (max_nmb_per_rank = {lim})" if lim else ""))

  # Function that builds an octree.
  @staticmethod
  def build_octree(regions, xmin, xmax, nroot, maxlev):
    """
    Builds an octree from the domain bounds 'xmin'/'xmax', the number of
    MeshBlocks on the root level, and the maximum refinement level of
    the AMR.

    Parameters:
    regions: Refinement regions determined by the respective criterion.
    xmin (float): Lower domain boundary.
    xmax (float): Upper domain boundary.
    nroot  (int): Number of MeshBlocks on the root level.
    maxlev (int): Maximum refinement level.
    """
    # Size of the MeshBlocks on the root level.
    w0 = (xmax - xmin) / nroot

    # Helper functions.
    def bounds(L, i, j, k):
      """Get the bounds of a MeshBlock on level 'L'."""
      w = w0 / 2**L
      lo = np.array([xmin + i*w, xmin + j*w, xmin + k*w])
      return lo, lo + w


    def target(key):
      """
      Get the target refinement level based on if MeshBlock fulfills
      any of the criteria in 'regions'.
      """
      lo, hi = bounds(*key)
      return max([r.level for r in regions if r.hits(lo, hi)], default=0)


    def children(L, i, j, k):
      """Get the children MeshBlocks after refinement."""
      return [(L+1, 2*i+a, 2*j+b, 2*k+c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


    def find(L, i, j, k):
      """Find neighboring blocks."""
      n = nroot * 2**L
      if not (0 <= i < n and 0 <= j < n and 0 <= k < n):
        return None
      for s in range(L + 1):
        key = (L - s, i >> s, j >> s, k >> s)
        if key in leaves:
          return key
      return None

    stack = list(itertools.product([0], range(nroot), range(nroot), range(nroot)))
    leaves = set()
    while stack:
      key = stack.pop()
      if key[0] < maxlev and key[0] < target(key):
        stack += children(*key)
      else:
        leaves.add(key)

    # 2:1 balance across faces, edges and corners
    while True:
      torefine = set()
      for (L, i, j, k) in leaves:
        for d in itertools.product((-1, 0, 1), repeat=3):
          if d != (0, 0, 0):
            nb = find(L, i+d[0], j+d[1], k+d[2])
            if nb is not None and nb[0] < L - 1:
              torefine.add(nb)
      if not torefine:
        return leaves
      for key in torefine:
        leaves.discard(key)
        leaves.update(children(*key))


# Define OriginRadius class.
class OriginRadius:
  """
  Determines whether minimum corner distance from origin is below some radius
  (needed for RefineRadii refinement).
  """
  def __init__(self, r, level):
    """
    OriginRadius initializer.

    Parameters:
    r   (float): Tracker refinement radius.
    level (int): Tracker 'reflevel'.
    """
    self.r, self.level = r, level

  # Whether a cell is a distance from the origin below the tracker radius.
  def hits(self, lo, hi):
    """
    Determines whether a cell is below the tracker radius and therefore
    flagged for refinement.

    Parameters:
    lo (float): Lower bound of the block.
    hi (float): Upper bound of the block.
    """
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    return np.min(np.sum(corners**2, axis=1)) < self.r**2


# Define Disk class.
class RefineDisk:
    """
    Class that determines closest point of a block to the axis
    R < rmax and |z| < zmax.
    """
    def __init__(self, c, rmax, zmax, level):
      """
      Initialize RefineDisk class.

      Parameters:
      c     (list): Current tracker position.
      rmax (float): Maximum radius of refined disk.
      zmax (float): Maximum height of refined disk.
      level  (int): Refinement level of refined disk.
      """
      self.c, self.rmax, self.zmax, self.level = np.asarray(c, float), rmax, zmax, level

    # Whether a MeshBlock with bounds will be refined by disk refinement criterion.
    def hits(self, lo, hi):
      """
      Determines whether a block with bounds 'lo' and 'hi'
      will be refined based on the disk refinement criterion.

      Parameters:
      lo (float): Lower bound of the block.
      hi (float): Upper bound of the block.
      """
      p = np.clip(0.0, lo - self.c, hi - self.c)
      return p[0]**2 + p[1]**2 < self.rmax**2 and abs(p[2]) < self.zmax

