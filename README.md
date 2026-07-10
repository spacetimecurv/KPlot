# KPlot
Interfacing plot-tools to create some plotting utilities for commonly used AthenaK output.

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
The packages are now built and can be used elsewhere.

## Usage
Currently supported are the following classes:
- ```HorizonFinder``` (reads in data from the AthenaK horizon finder)
- ```BatchMerger``` (concatenates batchtools ```output-*``` files)
- ```Waveform``` (reads in waveform data)
- ```SeriesPlot``` (reads in ```.bin``` files and creates a 2D plot series)
- ```History``` (reads in ```.hst``` files)
- ```Units``` (creates a unit conversion system for derived variables)
- ```SystemPlotter``` (full-run simulation visualizer spanning all ```output-XXXX``` restart segments)
Some examples outline the usage. Once built, the utilities can be called with:
```python
from kplot import *
```

### Full-run simulation visualization
```SystemPlotter``` combines every ```output-XXXX``` restart segment of an AthenaK
run and renders 1-D history/time-series plots plus parallel 2-D slice frames
(density, temperature, Y_e, radiation-M1 fields, neutrino energies) with optional
compact-object tracker, black-hole apparent-horizon and AMR-grid overlays.
Physical quantities are converted through ```kplot.athenak_units```
(```code```/```cgs```/```ngs```). All file names are keyed on the AthenaK job name
(```jobname```, default ```bhns```), so the same driver serves BNS, BHNS, BBH, etc.;
sections whose input files are absent are skipped. The slice plane is selectable
(```plane```: ```xy``` [default, auto-adds an xz companion panel], ```xz```, ```yz```)
and the time axis/title units can be ```Msun``` (default) or ```ms```
(```time_units```). Each frame series is written to its own
```<figpath>/<diagnostic>_<plane>/``` subfolder (e.g. ```Figs/temperature_xy/```),
and ```scripts/make_movies.sh``` builds one movie per folder into
```<figpath>/all_movies/```.

It can be driven programmatically:
```python
from kplot import SystemPlotter
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

A ready-to-edit driver (plus an ffmpeg movie-maker) lives in
[```scripts/```](scripts/): edit ```scripts/plot_system.sh``` (SIMPATH, JOBNAME,
PLANE, TIME_UNITS, SECTIONS, ...) and run it; it calls ```kplot-system``` and then
```scripts/make_movies.sh``` to assemble a movie per series into
```all_movies/```. Set ```DELETE_FRAMES=true``` in the driver to remove the
rendered frame PNGs afterwards (the movies are kept).
