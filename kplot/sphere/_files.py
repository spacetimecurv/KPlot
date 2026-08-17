"""Locating AthenaK spherical-surface (``file_type = sph``) VTK dumps.

AthenaK names a dump after the surfaces it holds:

    <job>.r=300.00.<var>.NNNNN.vtk          one surface at r = 300
    <job>.r=300.00-400.00.<var>.NNNNN.vtk   several surfaces spanning r = 300 .. 400
"""

import os
import re
from functools import lru_cache

import numpy as np

from athplot.load_sph_vtk import SphericalData

# A stored surface counts as the requested radius when it lies within this distance
# [M_sun].  File names carry two decimals, so anything looser would be guesswork.
RADIUS_TOL = 0.05


# ===================================================================
# File names and metadata
# ===================================================================

@lru_cache(maxsize=None)
def _name_pattern(jobname, variable):
    """Regex for ``<jobname>.r=RMIN[-RMAX].<variable>.NNNNN.vtk``."""
    num = r'\d+(?:\.\d+)?'
    return re.compile(rf'^{re.escape(jobname)}\.r=({num})(?:-({num}))?'
                      rf'\.{re.escape(variable)}\.(\d+)\.vtk$')


def parse_name(fname, jobname, variable):
    """Return ``(rmin, rmax, index)`` for a matching dump name, else None.

    Single-surface names give ``rmin == rmax``.
    """
    match = _name_pattern(jobname, variable).match(fname)
    if match is None:
        return None
    rmin = float(match.group(1))
    rmax = float(match.group(2)) if match.group(2) is not None else rmin
    return rmin, rmax, int(match.group(3))


def spec_str(rmin, rmax):
    """The ``r=...`` part of a dump name for the given radius range."""
    return f'r={rmin:.2f}' if rmin == rmax else f'r={rmin:.2f}-{rmax:.2f}'


def read_header(path, encoding='latin_1'):
    """Parse the ``key=value`` metadata AthenaK writes on the second line."""
    with open(path, 'r', encoding=encoding) as f:
        f.readline()
        meta = f.readline()
    header = {}
    for token in meta.split():
        key, sep, value = token.partition('=')
        if sep:
            header[key] = value
    return header


def read_time(path):
    """Simulation time [M_sun] of a dump, without parsing the field data."""
    return float(read_header(path)['time'])


def dump_nradii(path, rmin, rmax):
    """How many surfaces a dump holds, from its metadata line.

    The name only brackets them, so two dumps can share a name and still hold a
    different number of surfaces — a restart that changed ``nradii`` while keeping
    ``r_min`` and ``r_max`` does exactly that, and the surface wanted then sits at a
    different index in each.  A single-surface name needs no lookup at all.
    """
    if rmin == rmax:
        return 1
    nradii = read_header(path).get('nradii')
    # Dumps predating multiple radii carry no nradii; they are single-surface, and
    # the reader settles it either way.
    return int(nradii) if nradii is not None else SphericalData(path).nradii


def _stored_radii(path, rmin, rmax):
    """The radii a dump holds.

    A single-surface name pins the radius down already (to the two decimals the
    name carries), so only bundled dumps have to be opened.
    """
    if rmin == rmax:
        return np.array([rmin])
    return np.asarray(SphericalData(path).radii, dtype=float)


def _fmt_radii(radii):
    if len(radii) <= 4:
        return ', '.join(f'{r:.2f}' for r in radii)
    return ', '.join(f'{r:.2f}' for r in radii[:3]) + f', ..., {radii[-1]:.2f}'


# ===================================================================
# Snapshot lookup
# ===================================================================

class SnapshotSet:
    """The dumps of one variable that hold one requested extraction surface.

    A run may hold the same surface in more than one form — a restart that switched
    its ``<output>`` block writes single-surface dumps in the early segments and
    bundled ones later — so the surface index is kept per dump rather than globally.

    Attributes
    ----------
    variable : str
        The sph output variable, e.g. ``mhd_w_bcc`` or ``rad_m1_F``.
    radius : float
        Radius of the surface as stored in the files [M_sun].
    entries : dict
        ``{snapshot counter: (path, surface index)}``.
    groups : list
        ``(name spec, nradii, surface index, count)`` per dump form, for the log.
    """

    def __init__(self, variable, radius, entries, groups):
        self.variable = variable
        self.radius = radius
        self.entries = entries
        self.groups = groups

    @property
    def indices(self):
        """Snapshot counters, sorted."""
        return sorted(self.entries)

    @property
    def paths(self):
        """``{snapshot counter: path}``, dropping the surface index."""
        return {index: path for index, (path, _) in self.entries.items()}

    def shell(self, index):
        """(path, surface index) of snapshot `index`, ready to hand to :class:`Shell`."""
        return self.entries[index]

    def describe(self):
        """One-line summary for the analysis log."""
        line = f'{len(self.entries):5d} {self.variable} dumps at r = {self.radius:.2f}'
        if len(self.groups) == 1 and self.groups[0][1] == 1:
            return line
        detail = ', '.join(
            f'{spec}: surface {iradius} of {nradii} ({count})' if nradii > 1
            else f'{spec} ({count})'
            for spec, nradii, iradius, count in self.groups)
        return f'{line}  [{detail}]'

    def __len__(self):
        return len(self.entries)


def locate(sph_dirs, jobname, variable, radius, tol=RADIUS_TOL, hint=''):
    """Find the dumps of `variable` that hold the extraction surface at `radius`.

    Scans `sph_dirs` for ``<jobname>.r=....<variable>.NNNNN.vtk``, groups the dumps by
    the surfaces they hold and keeps every group with one at `radius`.  Both name forms
    are accepted, and a run that changed form across a restart still reads as one run,
    since each dump carries the index of the surface to use.

    Parameters
    ----------
    sph_dirs : list of str
        AthenaK ``output-XXXX/sph`` directories (all treated as one run).
    jobname, variable : str
        Job name and sph output variable the dumps are named after.
    radius : float
        Requested extraction radius [M_sun].
    tol : float
        How far a stored surface may sit from `radius` to still count as it [M_sun].
    hint : str
        Appended to the error message when nothing matches (e.g. which parfile
        output blocks the caller needs).

    Returns
    -------
    SnapshotSet

    Raises
    ------
    RuntimeError
        If no dump of `variable` exists, or none of them holds a surface at `radius`.
    """
    # Dumps are grouped by the surfaces they hold, so that one surface index is
    # valid for every dump of a group.
    groups = {}   # (rmin, rmax, nradii) -> {snapshot counter: path}
    for d in sph_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            parsed = parse_name(fname, jobname, variable)
            if parsed is None:
                continue
            rmin, rmax, index = parsed
            path = os.path.join(d, fname)
            key = (rmin, rmax, dump_nradii(path, rmin, rmax))
            groups.setdefault(key, {})[index] = path

    if not groups:
        raise RuntimeError(
            f'No {jobname}.r=*.{variable}.*.vtk dumps found in {list(sph_dirs)}.\n'
            f'Check the sph directories, the job name and the extraction radius.{hint}')

    matches, stored = [], {}
    for key, paths in sorted(groups.items()):
        rmin, rmax, nradii = key
        # The name brackets the surfaces inside, so a group that cannot hold the
        # radius is skipped without opening anything.
        if not (rmin - tol <= radius <= rmax + tol):
            continue
        radii = _stored_radii(paths[min(paths)], rmin, rmax)
        stored[key] = radii
        i = int(np.argmin(np.abs(radii - radius)))
        if abs(radii[i] - radius) <= tol:
            # Sort key: closest surface first, and among equally close ones the
            # smaller dump, so that is the one a shared counter resolves to.
            matches.append((abs(radii[i] - radius), nradii, key, i,
                            float(radii[i]), paths))

    if not matches:
        found = ', '.join(
            spec_str(rmin, rmax) + (f' (surfaces at {_fmt_radii(stored[key])})'
                                    if len(stored.get(key, ())) > 1 else '')
            for key in sorted(groups) for rmin, rmax, _ in (key,))
        raise RuntimeError(
            f'No {variable} surface at r = {radius:g} (within {tol:g} M_sun) in '
            f'{list(sph_dirs)}.\nThe dumps found cover {found}.{hint}')

    matches.sort(key=lambda m: m[:2])
    entries, group_info = {}, []
    for _, nradii, (rmin, rmax, _), iradius, _, paths in matches:
        added = 0
        for index, path in paths.items():
            # A counter shared by two groups keeps the closer surface.
            if index not in entries:
                entries[index] = (path, iradius)
                added += 1
        group_info.append((spec_str(rmin, rmax), nradii, iradius, added))
    return SnapshotSet(variable, matches[0][4], entries, group_info)


# ===================================================================
# Reading one surface
# ===================================================================

class Shell:
    """One extraction surface of an sph dump, however many surfaces the file holds.

    Exposes what the analyses need — ``time``, ``theta``, ``phi``, ``radius`` and
    ``shell(var)`` — with ``shell`` always returning the ``(Nphi, Ntheta)`` array of
    the selected surface, so single-radius and multi-radius dumps read alike.
    """

    def __init__(self, path, iradius=0):
        self._data = SphericalData(path)
        if not 0 <= iradius < self._data.nradii:
            raise IndexError(f'{os.path.basename(path)} holds '
                             f'{self._data.nradii} surface(s), no index {iradius}.')
        self._iradius = iradius
        self.path = path
        self.iradius = iradius
        self.nradii = self._data.nradii
        self.radius = float(self._data.radii[iradius])
        self.time = self._data.time
        self.cycle = self._data.cycle
        self.theta = self._data.theta
        self.phi = self._data.phi
        self.field_names = self._data.field_names

    def shell(self, var):
        """`var` on the selected surface, shape ``(Nphi, Ntheta)``."""
        return self._data.shell(var, self._iradius)
