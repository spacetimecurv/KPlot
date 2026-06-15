#########################################################################
# File: plot_series.py                                                  #
# Description: Plot series data from AthenaK bin output.                #
#########################################################################

# Import the series plot class.
from utilities import SeriesPlot

# Call the constructor.
series = SeriesPlot(
  series_path="/home/no96soq/athenak/runs/LEONARDO/basesetup/amrfix/bin",
  pattern="bhns.mhd_w_bcc_xy.*.bin",
  extent=[-50.0, 50.0, -50.0, 50.0],
  variable="dens",
  label=r"$\rho$",
  slice=None,
  lognorm=True,
  blackhole=True,
  output_dir="output/plot",
  tracker_path="/home/no96soq/athenak/runs/LEONARDO/basesetup/amrfix/bhns.co_0.txt",
  horizon_path="/home/no96soq/athenak/runs/LEONARDO/basesetup/amrfix/bhns.horizon_summary_0.txt")

# Plot and make a video.
series.plot()
series.video(framerate=4)
