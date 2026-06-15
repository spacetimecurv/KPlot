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
Some examples outline the usage. Once built, the utilities can be called with:
```python
from kplot import *
```
