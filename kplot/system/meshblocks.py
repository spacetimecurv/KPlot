"""Some MeshBlock utility targeted for AthenaK."""

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

  def __init__(self, parfile_path: str | None = None) -> None:
    """
    MeshBlock class initializer.

    Parameters:
    parfile_path (str): Path to a parfile holding information about
                        the simulation mesh
    """
    if parfile_path is not None:
      self.parfile = Parfile(parfile_path)

  # Plot 2D slice of current mesh, color-code is physical level.
  def plot_mesh(self, time: float | None = None, tracker_path: str | None = None,
                axis: str = "z", pos: float = 0.0, extent: float | None = None,
                cmap: str = "viridis", edgecolor: str = "k",
                linewidth: float = 0.3, savepath: str | None = None) -> None:
    """
    Plot 2D slice through MeshBlock octree. Static <refined_region*> blocks are
    always included. For adaptive refinement, tracker and radius regions are
    added; without 'tracker_path' the initial tracker positions co_N_x/y/z from
    the parfile are used, i.e. the grid at the start of the simulation.

    Parameters:
    time       (float): time at which to evaluate tracker positions.
    tracker_path (str): directory containing '*.co_[01].txt' tracker files
                        (initial positions from parfile if None).
    axis         (str): axis normal to the slice ('x', 'y', or 'z').
    pos        (float): position of the slice along 'axis'.
    extent     (float): half-width of plotted region.
    cmap         (str): colormap name.
    edgecolor    (str): color of MeshBlock edges.
    linewidth  (float): line width of the MeshBlock edges.
    savepath     (str): if given, save figure to this path, otherwise show it.

    Returns:
    None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap

    pf = self.parfile

    # Grid setup of base level.
    xmin, xmax = pf.get('mesh', 'x1min'), pf.get('mesh', 'x1max')
    nroot      = int(pf.get('mesh', 'nx1') // pf.get('meshblock', 'nx1'))

    # Refinement regions.
    regions = []
    for block in pf.parfile_dict:
      if block.startswith('refined_region'):
        regions.append(RefineBox([pf.get(block, f'x{d}min') for d in (1, 2, 3)],
                                 [pf.get(block, f'x{d}max') for d in (1, 2, 3)],
                                 pf.get(block, 'level', typ=int)))

    if pf.get('mesh_refinement', 'refinement', 'none', str) == 'adaptive':
      nco = pf.get('z4c', 'nco', 0, int)
      if nco > 0 and tracker_path is not None:
        if time is None:
          raise SystemExit("'time' is needed together with 'tracker_path'!")
        tracker_files = sorted(Path(tracker_path).glob("*.co_[01].txt"))
        if not 1 <= len(tracker_files) <= 2:
          raise SystemExit(f"Expected one or two tracker files under {tracker_path}, "
                           f"found {[f.name for f in tracker_files]}!")
        tracker = Tracker(*tracker_files)
      for ico in range(nco):
        if tracker_path is not None:
          c = tracker.tracker_pos(ico, float(time))
        else:
          c = [pf.get('z4c', f'co_{ico}_{d}', 0.0) for d in "xyz"]
        regions.append(RefineTracker(c, pf.get('z4c', f'co_{ico}_radius', 0.0),
                                        pf.get('z4c', f'co_{ico}_reflevel', -1, int)))

      for nr in range(16):
        if f'radius_{nr}_rad' not in pf.parfile_dict.get('z4c_amr', {}):
          break
        regions.append(OriginRadius(pf.get('z4c_amr', f'radius_{nr}_rad'),
                                    pf.get('z4c_amr', f'radius_{nr}_reflevel', -1, int)))

    maxlev = max([pf.get('mesh_refinement', 'num_levels', 1, int) - 1]
                 + [r.level for r in regions])

    # Build the octree.
    leaves = np.array(sorted(self.build_octree(regions, xmin, xmax, nroot, maxlev)))
    L, idx = leaves[:,0], leaves[:,1:]

    # Block widths and lower corners.
    w = (xmax - xmin) / nroot / 2.0**L
    lo = xmin + idx * w[:,None]

    # Select the blocks cut by slice.
    ax_n = {"x": 0, "y": 1, "z": 2}[axis]
    a, b = [d for d in range(3) if d != ax_n]
    mask = (lo[:,ax_n] <= pos) & (pos < lo[:,ax_n] + w)
    L, w, lo = L[mask], w[mask], lo[mask]

    # Rectangles as polygons.
    x0, y0 = lo[:,a], lo[:,b]
    verts  = np.stack([np.c_[x0, y0], np.c_[x0 + w, y0],
                       np.c_[x0 + w, y0 + w], np.c_[x0, y0 + w]], axis=1)

    # Discrete colormap.
    levels = np.arange(maxlev + 1)
    cm = ListedColormap(plt.get_cmap(cmap)(np.linspace(0, 1, maxlev + 1)))
    norm = BoundaryNorm(np.arange(-0.5, maxlev + 1.5), cm.N)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)

    coll = PolyCollection(verts, array=L, cmap=cm, norm=norm,
                          edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(coll)

    lim = (xmin, xmax) if extent is None else (-extent, extent)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    labels = "xyz"
    ax.set_xlabel(f"{labels[a]} [code units]")
    ax.set_ylabel(f"{labels[b]} [code units]")

    cbar = fig.colorbar(coll, ax=ax, ticks=levels)
    cbar.set_label("Physical refinement level")
    fig.tight_layout()
    if savepath is not None:
      fig.savefig(savepath, bbox_inches="tight")
      plt.close(fig)
    else:
      plt.show()

  # Refinement function that predicts number of MeshBlocks based on
  # refinement criterion (wrapper).
  def predict_meshblocks(self, mode: str = "disk", **refargs) -> None:
    """
    Wrapper function predicting the number of MeshBlocks upon refinement,
    based on the refinement criterion used, and the underlying mesh from the parfile.

    Parameters:
    mode (str): Which refinement criterion is used.
    **refargs: Refinement arguments passed to the refinement function.

    Returns:
    None.
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
  def __predict_meshblocks_disk(self, **kwargs) -> None:
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

    Returns:
    None.
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

    if all(k in kwargs for k in ("nodes", "gpus-per-node")):
      nr = kwargs["nodes"] * kwargs["gpus-per-node"]
      lim = pf.get('mesh_refinement', 'max_nmb_per_rank', 0, int)
      print(f"\n{nr} ranks -> {len(leaves)/nr:.1f} blocks/rank"
            + (f" (max_nmb_per_rank = {lim})" if lim else ""))

  # Function that builds an octree.
  @staticmethod
  def build_octree(regions: list, xmin: float, xmax: float, nroot: int,
                   maxlev: int) -> set[tuple[int, int, int, int]]:
    """
    Builds an octree from the domain bounds 'xmin'/'xmax', the number of
    MeshBlocks on the root level, and the maximum refinement level of
    the AMR.

    Parameters:
    regions (list): Refinement regions, each with a hits(lo, hi) method and a level.
    xmin (float): Lower domain boundary.
    xmax (float): Upper domain boundary.
    nroot  (int): Number of MeshBlocks on the root level.
    maxlev (int): Maximum refinement level.

    Returns:
    Set of leaf MeshBlocks as (level, i, j, k).
    """
    # Size of the MeshBlocks on the root level.
    w0 = (xmax - xmin) / nroot

    # Helper functions.
    def bounds(L: int, i: int, j: int, k: int) -> tuple[np.ndarray, np.ndarray]:
      """
      Get the bounds of a MeshBlock on level 'L'.

      Parameters:
      L (int): Refinement level.
      i, j, k (int): Logical location on that level.

      Returns:
      (lower corner, upper corner).
      """
      w = w0 / 2**L
      lo = np.array([xmin + i*w, xmin + j*w, xmin + k*w])
      return lo, lo + w


    def target(key: tuple[int, int, int, int]) -> int:
      """
      Get the target refinement level based on if MeshBlock fulfills
      any of the criteria in 'regions'.

      Parameters:
      key (tuple): MeshBlock as (level, i, j, k).

      Returns:
      Highest level of the regions hitting the MeshBlock (0 if none).
      """
      lo, hi = bounds(*key)
      return max([r.level for r in regions if r.hits(lo, hi)], default=0)


    def children(L: int, i: int, j: int, k: int) -> list[tuple[int, int, int, int]]:
      """
      Get the children MeshBlocks after refinement.

      Parameters:
      L (int): Refinement level.
      i, j, k (int): Logical location on that level.

      Returns:
      The eight children as (level, i, j, k).
      """
      return [(L+1, 2*i+a, 2*j+b, 2*k+c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


    def find(L: int, i: int, j: int, k: int) -> tuple[int, int, int, int] | None:
      """
      Find the leaf covering a (possibly neighboring) MeshBlock location.

      Parameters:
      L (int): Refinement level.
      i, j, k (int): Logical location on that level.

      Returns:
      The covering leaf as (level, i, j, k), or None if outside the domain.
      """
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
  def __init__(self, r: float, level: int) -> None:
    """
    OriginRadius initializer.

    Parameters:
    r   (float): Tracker refinement radius.
    level (int): Tracker 'reflevel'.
    """
    self.r, self.level = r, level

  # Whether a cell is a distance from the origin below the tracker radius.
  def hits(self, lo: np.ndarray, hi: np.ndarray) -> bool:
    """
    Determines whether a cell is below the tracker radius and therefore
    flagged for refinement.

    Parameters:
    lo (np.ndarray): Lower corner of the block.
    hi (np.ndarray): Upper corner of the block.

    Returns:
    True if the block is refined.
    """
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    return np.min(np.sum(corners**2, axis=1)) < self.r**2


# Define Disk class.
class RefineDisk:
    """
    Class that determines closest point of a block to the axis
    R < rmax and |z| < zmax.
    """
    def __init__(self, c: list, rmax: float, zmax: float, level: int) -> None:
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
    def hits(self, lo: np.ndarray, hi: np.ndarray) -> bool:
      """
      Determines whether a block with bounds 'lo' and 'hi'
      will be refined based on the disk refinement criterion.

      Parameters:
      lo (np.ndarray): Lower corner of the block.
      hi (np.ndarray): Upper corner of the block.

      Returns:
      True if the block is refined.
      """
      p = np.clip(0.0, lo - self.c, hi - self.c)
      return p[0]**2 + p[1]**2 < self.rmax**2 and abs(p[2]) < self.zmax



class RefineBox:
  """
  Static refinement region from a <refined_region*> block of the parfile.
  """
  def __init__(self, lo: list, hi: list, level: int) -> None:
    """
    Initialize RefineBox class.

    Parameters:
    lo  (list): Lower corner (x1min, x2min, x3min) of the region.
    hi  (list): Upper corner (x1max, x2max, x3max) of the region.
    level (int): Physical refinement level of the region.
    """
    self.lo, self.hi, self.level = np.asarray(lo, float), np.asarray(hi, float), level

  def hits(self, lo: np.ndarray, hi: np.ndarray) -> bool:
    """
    Determines whether a block with bounds 'lo' and 'hi' overlaps the region,
    following the index selection in AthenaK's build_tree.cpp.

    Parameters:
    lo (np.ndarray): Lower corner of the block.
    hi (np.ndarray): Upper corner of the block.

    Returns:
    True if the block is refined.
    """
    return bool(np.all(hi > self.lo) and np.all(lo < self.hi))
