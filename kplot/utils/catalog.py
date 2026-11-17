"""
Field-based lookup of AthenaK dumps.

Analyses ask for the fields they need (e.g. 'dens', 'z4c_alpha', 'E:0'), not for output
block names, so it does not matter whether the lapse comes from a `z4c_alpha` block or a
`z4c_gauge` block, or whether the M1 moments are split into `rad_m1_E/N/F`. Dumps of
different blocks are matched by their cycle.
"""

# Built-in libraries.
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Third-party libraries.
import numpy as np

# KPlot utility.
from kplot.system.checkpoints import find_segments
from .bin3d import read_bin_header
from .sph import RADIUS_TOL, read_header, read_layout

HYDRO    = ('dens', 'velx', 'vely', 'velz', 'press', 's_00', 'temperature')
MAGNETIC = ('bcc1', 'bcc2', 'bcc3')
METRIC   = ('adm_gxx', 'adm_gxy', 'adm_gxz', 'adm_gyy', 'adm_gyz', 'adm_gzz')
GAUGE    = ('z4c_alpha', 'z4c_betax', 'z4c_betay', 'z4c_betaz')

_DUMP    = re.compile(r'^(?P<stem>.+)\.(?P<num>\d+)\.(?P<ext>bin|vtk)$')
_IGNORE  = {'weights'}
_THREADS = 32


@dataclass
class Stream:
  """
  The dumps of one output block in one directory.

  Attributes:
  stem   (str): file name without the dump counter, e.g. 'bhns.r=5.00-500.00.z4c_gauge'.
  kind   (str): 'bin' or 'sph'.
  files (dict): {counter: path}.
  fields (tuple): fields every dump holds.
  plane  (str): bin only; None for 3D, else 'xy', 'xz' or 'yz'.
  radii (np.ndarray): sph only; radii of the surfaces.
  """
  stem: str
  kind: str
  files: dict
  fields: tuple
  plane: str | None = None
  radii: np.ndarray | None = None
  _stamps: dict = field(default=None, repr=False)

  @property
  def name(self) -> str:
    return self.stem.split('.')[-1]

  def stamps(self) -> dict:
    """
    {counter: (cycle, time)} of every dump, read from the headers once.
    """
    if self._stamps is None:
      read = _bin_stamp if self.kind == 'bin' else _sph_stamp
      with ThreadPoolExecutor(_THREADS) as ex:
        self._stamps = dict(zip(self.files, ex.map(read, self.files.values())))
    return self._stamps


@dataclass
class Snapshot:
  """
  One output time with the dump (and surface index) providing each requested field.

  Attributes:
  time  (float): simulation time [M_sun].
  cycle   (int): simulation cycle.
  tag     (str): dump counter of the first requested field, for output file names.
  sources (dict): {field: (path, surface index or None)}.
  """
  time: float
  cycle: int
  tag: str
  sources: dict

  def by_path(self) -> dict:
    """
    {path: [fields read from it]} in request order.
    """
    out = {}
    for name, (path, _) in self.sources.items():
      out.setdefault(path, []).append(name)
    return out


def _bin_stamp(path: str) -> tuple:
  h = read_bin_header(path)
  return h['cycle'], h['time']


def _sph_stamp(path: str) -> tuple:
  h = read_header(path)
  return int(h.get('cycle', -1)), float(h['time'])


def dump_dirs(simpath: str, kind: str, batchtools: bool = True) -> list[str]:
  """
  Directories holding the dumps of a run, in segment order.

  Parameters:
  simpath     (str): simulation directory.
  kind        (str): 'bin' or 'sph'.
  batchtools (bool): whether simpath holds output-XXXX segments.

  Returns:
  list of directories (<segment>/<kind>, or <segment> itself where that does not exist
  for bin dumps or without batchtools).
  """
  out = []
  for seg in find_segments(simpath, batchtools):
    sub = os.path.join(seg, kind)
    if os.path.isdir(sub):
      out.append(sub)
    elif kind == 'bin' or not batchtools:
      out.append(seg)
  return out


def scan(dirs: list[str], kind: str, jobname: str | None = None) -> list[Stream]:
  """
  Group the dumps in `dirs` into streams and read their fields from one dump each.

  Parameters:
  dirs    (list): directories in segment order (later segments win).
  kind     (str): 'bin' or 'sph'.
  jobname  (str): only consider dumps named <jobname>.*; None for all.

  Returns:
  list of Stream.
  """
  ext = 'bin' if kind == 'bin' else 'vtk'
  groups = []
  for d in dirs:
    if not os.path.isdir(d):
      continue
    by_stem = {}
    for fname in os.listdir(d):
      m = _DUMP.match(fname)
      if m is None or m['ext'] != ext:
        continue
      if jobname is not None and not fname.startswith(f'{jobname}.'):
        continue
      by_stem.setdefault(m['stem'], {})[int(m['num'])] = os.path.join(d, fname)
    groups += [(stem, files) for stem, files in sorted(by_stem.items())]

  def build(group: tuple) -> Stream:
    stem, files = group
    probe = files[max(files)]
    if kind == 'bin':
      h = read_bin_header(probe)
      return Stream(stem, kind, files, tuple(f for f in h['fields'] if f not in _IGNORE),
                    plane=h['plane'])
    lay = read_layout(probe)
    radii = np.fromfile(probe, dtype='>f4', count=lay['dims'][0], offset=lay['radii']) \
      if lay['radii'] is not None else _radii_from_points(probe, lay)
    return Stream(stem, kind, files, tuple(f for f in lay['fields'] if f not in _IGNORE),
                  radii=radii.astype(float))

  with ThreadPoolExecutor(_THREADS) as ex:
    return list(ex.map(build, groups))


def _radii_from_points(path: str, lay: dict) -> np.ndarray:
  nr = lay['dims'][0]
  return np.fromfile(path, dtype='>f4', count=3 * nr, offset=lay['points'])[0::3]


def find(streams: list[Stream], fields: tuple, *, radius: float | None = None,
         volume: bool = False, plane: str | None = None, nearest: tuple = (),
         tol: float = 4.0, rtol: float = RADIUS_TOL, verbose: bool = True) -> list[Snapshot]:
  """
  Every output time at which all `fields` are available.

  Parameters:
  streams (list): streams from scan().
  fields (tuple): field names; the first one sets Snapshot.tag.
  radius (float): sph surface radius [M_sun]; selects the closest stored surface.
  volume  (bool): sph volume: all fields from streams sharing one radial grid.
  plane    (str): bin slice plane ('xy', 'xz', 'yz'); None for 3D dumps.
  nearest (tuple): fields matched to the closest time within `tol` instead of the cycle.
  tol    (float): time tolerance for `nearest` [M_sun].
  rtol   (float): radius tolerance [M_sun].
  verbose (bool): print which streams provide the fields.

  Returns:
  list of Snapshot, sorted by time.
  """
  fields = tuple(dict.fromkeys(fields))
  exact = [f for f in fields if f not in nearest]
  if not exact:
    raise ValueError('At least one field has to be matched by cycle.')
  cands = _candidates(streams, set(fields), radius, volume, plane, rtol)
  if verbose:
    _report(cands, fields, radius)

  by_cycle, by_time = {}, {f: [] for f in nearest}
  for s in cands:
    rank, ir = _rank(s, radius)
    for num, (cycle, time) in s.stamps().items():
      entry = (rank, s.files[num], ir, time, num)
      slot = by_cycle.setdefault(cycle, {})
      for f in exact:
        if f in s.fields and (f not in slot or rank <= slot[f][0]):
          slot[f] = entry
      for f in nearest:
        if f in s.fields:
          by_time[f].append(entry)
  for entries in by_time.values():
    entries.sort(key=lambda e: e[3])

  snaps, skipped = [], 0
  for cycle, slot in by_cycle.items():
    sources = {f: slot[f][1:3] for f in exact if f in slot}
    time = slot[exact[0]][3] if exact[0] in slot else None
    for f in nearest:
      entry = _closest(by_time[f], time, tol) if time is not None else None
      if entry is not None:
        sources[f] = entry[1:3]
    if len(sources) < len(fields):
      skipped += 1
      continue
    snaps.append(Snapshot(time, cycle, f'{slot[exact[0]][4]:05d}',
                          {f: sources[f] for f in fields}))

  if verbose:
    if skipped:
      print(f'  [skip] {skipped} output times missing one of the fields')
    print(f'Found {len(snaps)} snapshots.')
  return sorted(snaps, key=lambda s: s.time)


def _candidates(streams: list[Stream], wanted: set, radius: float | None, volume: bool,
                plane: str | None, rtol: float) -> list[Stream]:
  if radius is not None or volume:
    cands = [s for s in streams if s.kind == 'sph' and wanted & set(s.fields)]
  else:
    cands = [s for s in streams if s.kind == 'bin' and s.plane == plane
             and wanted & set(s.fields)]
  if radius is not None:
    cands = [s for s in cands if _rank(s, radius)[0][0] <= rtol]
  if volume:
    cands = _volume_streams(cands, wanted)
  missing = wanted - {f for s in cands for f in s.fields}
  if missing:
    raise LookupError(_missing_message(streams, missing, radius, volume, plane))
  return cands


def _rank(s: Stream, radius: float | None) -> tuple:
  """
  ((distance to radius, number of radii), surface index); lower ranks are preferred.
  """
  if radius is None:
    return (0.0, 0), None
  i = int(np.argmin(np.abs(s.radii - radius)))
  return (abs(s.radii[i] - radius), len(s.radii)), i


def _closest(entries: list, time: float, tol: float) -> tuple | None:
  if not entries:
    return None
  i = int(np.argmin([abs(e[3] - time) for e in entries]))
  return entries[i] if abs(entries[i][3] - time) <= tol else None


def _report(cands: list[Stream], fields: tuple, radius: float | None) -> None:
  counts = {}
  for s in cands:
    _, ir = _rank(s, radius)
    where = f' r={s.radii[ir]:.2f}' if ir is not None else ''
    key = f'{s.stem}{where}: {_short([f for f in fields if f in s.fields])}'
    counts[key] = counts.get(key, 0) + len(s.files)
  for key, n in counts.items():
    print(f'  {n:5d} x {key}')


def _volume_streams(cands: list[Stream], wanted: set) -> list[Stream]:
  grids = {}
  for s in cands:
    if len(s.radii) > 1:
      grids.setdefault(tuple(np.round(s.radii, 4)), []).append(s)
  complete = [g for g in grids.values() if wanted <= {f for s in g for f in s.fields}]
  if not complete:
    return []
  return max(complete, key=lambda g: (len(g[0].radii), sum(len(s.files) for s in g)))


def _short(names: list) -> str:
  return ', '.join(names) if len(names) <= 6 else ', '.join(names[:5]) + f', ... ({len(names)})'


def _missing_message(streams: list, missing: set, radius: float | None, volume: bool,
                     plane: str | None) -> str:
  where = (f'at r = {radius:g}' if radius is not None else 'on a common radial grid'
           if volume else f'in {plane} slices' if plane else 'in 3D bin dumps')
  have = sorted({f for s in streams for f in s.fields})
  return (f'No dumps provide {", ".join(sorted(missing))} {where}.\n'
          f'Fields found: {_short(have)}\nStreams: '
          + ', '.join(sorted({s.stem for s in streams})[:20]))
