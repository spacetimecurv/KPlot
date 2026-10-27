#########################################################################
# File: analyze_meshblocks.py                                           #
# Description: Meshblock utility demonstration.                         #
#########################################################################

# Import MeshBlock class.
from kplot.system.meshblocks import MeshBlock

# Predict how many MeshBlocks will be generated, when we refine
# within a radius R and height z of certain target level.
refargs = {
  "disk-time": 2200.0,
  "disk-reflevel": 10,
  "disk-rmax": 50.0,
  "disk-zmax": 10.0,
  "disk-tracker": 0,
  "disk-tracker-path": "/home/no96soq/athenak/runs/PhysicsComparisonBHNS/MHD_M1_120_MRI/tracker",
  "nodes": 64,
  "gpus-per-node": 4
}

mb = MeshBlock(parfile_path="/home/no96soq/athenak/runs/PhysicsComparisonBHNS/MHD_M1_120_MRI/parfile.par")
mb.predict_meshblocks(mode="disk", **refargs)