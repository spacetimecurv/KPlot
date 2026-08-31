#########################################################################
# File: analyze_sphere.py                                               #
# Description: Ejecta + neutrino analysis of AthenaK sph output.        #
#########################################################################

# Analysis of AthenaK's spherical-surface extraction output (file_type = sph):
# what crosses the extraction sphere.
import glob

from kplot.sphere import ejecta, mergertime, neutrinos, plots, poynting

SIMPATH = "/path/to/sim"          # parent dir holding the output-XXXX segments
EOS_TABLE = "/path/to/DD2.h5"     # PyCompOSE HDF5 table matching the simulation EOS
OUTPUT_DIR = f"{SIMPATH}/analysis"
RADIUS = 300.0                    # sph extraction radius [M_sun]
JOBNAME = "bns"                   # AthenaK job name prefixing the VTK files

# All output-XXXX/sph segments are analysed together as one run.
sph_dirs = sorted(glob.glob(f"{SIMPATH}/output-*/sph"))

# --- Step 0: merger time from the peak of |psi4_22|^2 of the GW (2,2) mode ---
# The ejecta window and the plots are all referenced to this time.
t_merger, _ = mergertime.merger_time(SIMPATH, radius=RADIUS)
mergertime.write_merger_time(f"{OUTPUT_DIR}/merger_time.txt", t_merger, RADIUS)

# --- Step 1: ejecta (mass flux, v_inf / theta / Y_e distributions) ---
# Only the first T_POST_MS after the merger enter the 2-D histogram; the
# Mej_rate time series always covers every snapshot.
T_POST_MS = 25.0
t_stop = t_merger + T_POST_MS * 1e-3 / ejecta.MSUN_TO_S
ejecta.analyze(
  sph_dirs,
  EOS_TABLE,
  OUTPUT_DIR,
  radius=RADIUS,
  jobname=JOBNAME,
  dfloor=3e-15,      # density floor used in the simulation [code units]
  t_stop=t_stop,
  n_workers=8,       # login node: keep <= 4; compute node: the core count
)

# --- Step 2: neutrinos (M1 luminosities + mean energies per species) ---
# Needs rad_m1_E / rad_m1_F / rad_m1_N sph output at the same radius.
neutrinos.analyze(
  sph_dirs,
  OUTPUT_DIR,
  radius=RADIUS,
  jobname=JOBNAME,
  n_workers=8,
)

# --- Step 3: Poynting flux ---
poynting.analyze(
  sph_dirs,
  OUTPUT_DIR,
  radius=RADIUS,
  jobname=JOBNAME,
  n_workers=8,
)

# --- Step 4: summary figures -> fig_ejecta.pdf, fig_neutrino.pdf, ... ---
plots.main([
  "--output-dir", OUTPUT_DIR,
  "--t-merger", str(t_merger),
  "--radius", str(RADIUS),
  "--from-merger",     # x-axis as t - t_merger [ms] instead of absolute time
  "--poynting",
  "--nprocs", 8,
])
