# KPlot
Interfacing plot-tools to create some plotting utilities for commonly used AthenaK output.

## Credits
The spherical-surface analysis in ```kplot.sphere``` and ```scripts/sphere/```
(ejecta, neutrino and merger-time diagnostics) is taken from
[AthenaK_BNS_visualization](https://github.com/yiqiu0714/AthenaK_BNS_visualization). The full-run plotter ```kplot.system``` is likewise adapted from that repository's ```BNS_all.py```.
```kplot``` also builds on [plot-tools](https://github.com/jfields7/plot-tools)
(```athplot```), imported as a submodule in ```external/```.

## Build
We first clone the repository to the local workstation with:
```bash
git clone https://github.com/spacetimecurv/KPlot.git
```
Then we clone the plot-tools submodule with:
```bash
git submodule update --init --recursive
```
Next, we built plot-tools followed by KPlot:
```bash
pip install -e external/plot-tools
pip install -e .
```
The packages are now built and can be used elsewhere. It is recommended to do this in a separate
environment to keep it from the global PATH.

## Usage
Currently supported are the following classes:
- ```HorizonFinder``` (reads in data from the AthenaK horizon finder)
- ```BatchMerger``` (concatenates batchtools ```output-*``` files)
- ```Waveform``` (reads in waveform data)
- ```SeriesPlot``` (reads in ```.bin``` files and creates a 2D plot series)
- ```History``` (reads in ```.hst``` files)
- ```Units``` (creates a unit conversion system for derived variables)
- ```SystemPlotter``` (full-run simulation visualizer spanning all ```output-XXXX``` restart segments)
- ```kplot.sphere``` (ejecta, neutrino and merger-time analysis of spherical-surface extraction output)
- ```kplot.volume``` (post-merger disk diagnostics from 3D volume-domain snapshots)

Some examples outline the usage. Once built, the utilities can be called with, e.g.:
```python
from kplot.system.plotter import SystemPlotter
from kplot.system.horizon import HorizonFinder
```

Each workflow has a shell driver (and there are also selected examples for usage):

| Workflow | Example | Driver |
|---|---|---|
| Full-run visualization | [```examples/plot_system.py```](examples/plot_system.py) | [```scripts/system/plot_system.sh```](scripts/system/plot_system.sh) |
| Ejecta + neutrino analysis | [```examples/analyze_sphere.py```](examples/analyze_sphere.py) | [```scripts/sphere/run_analysis.sh```](scripts/sphere/run_analysis.sh) |
| Post-merger disk analysis | *(none yet)* | [```scripts/volume/run_analysis.sh```](scripts/volume/run_analysis.sh) |

### Repository layout
The package is split into three subpackages, each with a matching driver folder in
```scripts/```:

```
kplot/              the installed Python package
  system/           whole-run diagnostics
                      plotter.py (SystemPlotter), history.py, horizon.py,
                      waveforms.py, seriesplot.py, batchmerge.py,
                      units.py, athenak_units.py, bin_convert.py
  sphere/           spherical-surface (sph) extraction analysis
                      ejecta.py, neutrinos.py, mergertime.py, poynting.py,
                      butterfly.py, accretion.py, plots.py, comparison.py
  volume/           3D volume-domain (bin) analysis
                      disk.py, plots.py
scripts/            ready-to-edit shell drivers, one folder per workflow
  system/           plot_system.sh, make_movies.sh                  -> kplot.system
  sphere/           run_analysis.sh, run_comparison.sh, config.ini  -> kplot.sphere
  volume/           run_analysis.sh, config.ini                     -> kplot.volume
examples/           small, self-contained usage examples
external/           plot-tools submodule (athplot)
```

```kplot.system``` covers the simulation as a whole (history files, trackers,
horizons, waveforms, slice series); ```kplot.sphere``` covers what crosses a
spherical extraction surface (ejecta, neutrinos); ```kplot.volume``` covers the
3D volume domain (currently the post-merger disk). IMPORTANT: the ```kplot.system``` and
```kplot.sphere``` modules require the data to be stored inside batchtools like arrangement, i.e.
in folders with the signature ```output-XXXX```. If this is not the case, then one segment has to be created and the data has to be moved there manually such that these modules can be found. For
```kplot.volume``` this currently is not a requirement, it is even recommended to store the 3D data
in one directory.

### Full-run simulation visualization
```SystemPlotter``` combines every batchtools ```output-XXXX``` restart segment (see note above) of an AthenaK run and renders 1-D history/time-series plots plus parallel 2-D slice frames
(density, temperature, Y_e, radiation-M1 fields, neutrino energies) with optional
compact-object tracker, black-hole apparent-horizon and AMR-grid overlays.
Physical quantities are converted through ```kplot.athenak_units```
(```code```/```cgs```/```ngs```). The slice plane is selectable
(```plane```: ```xy```, ```xz```, ```yz```)
and the time axis/title units can be ```Msun``` (default) or ```ms```
(```time_units```). Each frame series is written to its own
```<figpath>/<diagnostic>_<plane>/``` subfolder (e.g. ```Figs/temperature_xy/```),
and ```scripts/system/make_movies.sh``` builds one movie per folder into
```<figpath>/all_movies/```.

It can be driven by the following python command:
```python
from kplot.system.plotter import SystemPlotter
SystemPlotter(simpath="/path/to/sim", units="cgs", jobname="bhns",
              plane="xy", show_horizon=True).run(["density", "history"])
```
or from the command line:
```bash
python3 -m kplot.system --simpath /path/to/sim --units cgs \
    --jobname bhns --plane xy --show-horizon --sections density history
# or, via the installed console script:
kplot-system --simpath /path/to/sim --units cgs --sections density history
```

A automated script lives in
[```scripts/system/```](scripts/system/). Configuration goes through ```scripts/system/config.example.ini```:
```bash
cd scripts/system
cp config.example.ini config.ini    # set parameters
bash plot_system.sh
```
Make sure to modify the ```config.ini``` file, before running the bash script, since
otherwise default values will be applied.

### Spherical-surface analysis (ejecta + neutrinos)
```kplot.sphere``` analyses AthenaK's spherical-surface extraction output
(```file_type = sph```) — what crosses the extraction sphere, spanning all
batchtools ```output-XXXX``` restart segments:
- ```kplot.sphere.mergertime``` — merger time from max ```|psi4_22|^2``` of the GW (2,2) mode
- ```kplot.sphere.ejecta``` — ejecta mass flux, ```v_inf```/```theta```/```Y_e``` distributions
  (geodesic and Bernoulli criteria; needs a PyCompOSE HDF5 EOS table)
- ```kplot.sphere.neutrinos``` — M1 neutrino luminosities and flux-weighted mean
  energies per species (```rad_m1_E/F/N``` output)
- ```kplot.sphere.poynting``` — sphere-integrated Poynting luminosity and its
  angular map vs time
- ```kplot.sphere.butterfly``` — density-weighted, azimuthally-averaged toroidal
  field vs theta and time (dynamo polarity reversals, cf. arXiv:2211.07158)
- ```kplot.sphere.plots``` / ```kplot.sphere.comparison``` — summary figures
  (```fig_ejecta```, ```fig_neutrino```, ```fig_butterfly```, ...), and
  multi-model overlays
- ```kplot.sphere.accretion``` — post-merger accretion rate from baryon bookkeeping
Also here a pipeline exists under [```scripts/sphere/```](scripts/sphere/):
```bash
cd scripts/sphere
cp config.example.ini config.ini    # set parameteres
bash run_analysis.sh                # merger time + ejecta + neutrino + plots
bash run_analysis.sh --ejecta       # or run individual steps
```

Each step is also an installed console script, so it can be run directly:
```bash
# merger time (writes merger_time.txt, read back by the other steps)
kplot-sphere-mergertime --simpath /path/to/sim --radius 300 \
    --out /path/to/sim/analysis/merger_time.txt

# ejecta: pass --sph-dir once per restart segment
kplot-sphere-ejecta --sph-dir /path/to/sim/output-0000/sph \
                    --sph-dir /path/to/sim/output-0001/sph \
    --eos-table DD2.h5 --output-dir /path/to/sim/analysis \
    --radius 300 --jobname bns --t-merger 3122.5 --t-post-ms 25

# neutrinos (needs rad_m1_E/F/N sph output at the same radius)
kplot-sphere-neutrinos --sph-dir /path/to/sim/output-0000/sph \
    --output-dir /path/to/sim/analysis --radius 300

# figures: fig_ejecta.pdf + fig_neutrino.pdf
kplot-sphere-plot --output-dir /path/to/sim/analysis --t-merger 3122.5 \
    --radius 300 --from-merger
```
or driven as a library — see [```examples/analyze_sphere.py```](examples/analyze_sphere.py):
```python
import glob
from kplot.sphere import ejecta, mergertime, neutrinos

simpath, radius = "/path/to/sim", 300.0
sph_dirs = sorted(glob.glob(f"{simpath}/output-*/sph"))

# merger time from the peak of |psi4_22|^2 of the GW (2,2) mode
t_merger, _ = mergertime.merger_time(simpath, radius=radius)

# ejecta: only the first 25 ms after merger enter the 2-D histogram
t_stop = t_merger + 25.0 * 1e-3 / ejecta.MSUN_TO_S
ejecta.analyze(sph_dirs, "DD2.h5", f"{simpath}/analysis",
               radius=radius, jobname="bns", t_stop=t_stop, n_workers=8)

neutrinos.analyze(sph_dirs, f"{simpath}/analysis", radius=radius, n_workers=8)
```

Outputs land in ```<simpath>/analysis/``` as plain text (plus ```.npy``` for the 2-D
```v_inf```–```theta``` histograms), so they can be re-plotted or compared later
without re-reading the VTK files.

### Post-merger disk analysis (volume)
```kplot.volume``` analyses AthenaK's 3D volume-domain output (```file_type = bin```)
directly, rather than a spherical extraction surface:
- ```kplot.volume.disk``` — per-snapshot post-merger disk diagnostics (mass,
  angular momentum, MRI quality factor, ...) from ```mhd_w_bcc```/```adm```/```z4c_alpha```/
  ```z4c_betax/y/z``` 3D snapshots, an EOS table and a tracker/horizon file
  (or a manual center) to locate the compact object
- ```kplot.volume.plots``` — one histogram/profile frame per snapshot (```Ye```,
  entropy, temperature histograms; ```Sigma```, ```rho```, ```Ye```, ```T```,
  ```H/R```, ```Omega``` radial profiles), with fixed, log-scaled axes so
  ```make_movies.sh``` can turn them into an evolution movie

The driver lives in [```scripts/volume/```](scripts/volume/):
```bash
cd scripts/volume
cp config.example.ini config.ini    # set bindir + eos_table for your machine
bash run_analysis.sh                # disk analysis + plots + movies
bash run_analysis.sh --disk         # or run individual steps
bash run_analysis.sh --plot         # re-plot without re-running the analysis
```
Results are written to ```<bindir>/disk/```:
```
scalars/disk_scalars_<snap>.json               disk mass, angular momentum, MRI Q_z, ...
profiles/disk_profiles_<snap>.csv              radial profiles (Sigma, rho, Ye, T, H/R, Omega, ...)
histograms/disk_histograms_<snap>.csv          Ye/entropy/temperature histograms
frames/{histograms,profiles,scalars}/*.png     one frame per snapshot
frames/all_movies/*.mp4                        stitched by scripts/system/make_movies.sh
```
