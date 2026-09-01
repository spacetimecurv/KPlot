"""
Spectra analysis for a post-merger disk from an AthenaK BHNS 3D snapshot.

The following data is required for this:
<output1>
file_type = bin
variable  = mhd_w_bcc
dt        = 500.0
id        = mhd_w_bcc_3D
"""

# Built-in libraries.
import os
from multiprocessing import Pool, cpu_count
import argparse
import re
from tqdm import tqdm

# Third-party libraries.
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# KPlot utility.
from kplot.system.athenak_units import G, c, Msun
from kplot.volume.disk import find_snapshots, block_cell_geometry, load_tracker
from kplot.system.bin_convert import read_binary

# Manual conversion between code and physical units.
ms    = 4.925490949141889e-6 * 1e3
km    = 1.47662506140
gcm3  = 6.175828283964599e+17
Gauss = G**(-1.5) * Msun**(-1) * c**4 * np.sqrt(4 * np.pi)

# Per-worker state.
_W_TRACKER = None


def _init_spectra_worker(tracker):
  """Pool initializer: load the tracker file."""
  global _W_TRACKER
  _W_TRACKER = tracker


def _process_snapshot_spectra(args):
  """Per-snapshot worker for spectra analysis."""
  # Unpack arguments.
  (mhd_file, target_dx, window_rad, outdir, center) = args

  # Read the data.
  snapshot = mhd_file.split("/")[-1].split(".")[-2].strip()
  data_mhd = read_binary(mhd_file)
  time     = data_mhd['time']
  x1, x2, x3, _, dx, _, _ = block_cell_geometry(data_mhd) # CC-coordinates and volume for each MB.

  # Get the center of the remnant (the box drifts with it).
  if _W_TRACKER is not None:
    tracker     = _W_TRACKER
    tracker_idx = np.argmin(np.abs(tracker["time"] - time))
    center      = np.array([tracker["x"][tracker_idx],
                            tracker["y"][tracker_idx],
                            tracker["z"][tracker_idx]])
  elif _W_TRACKER is None and center:
    center = np.array(center)
  else:
    raise SystemExit("Either tracker file or manual center has to be specified!")

  # Primitives.
  rho = np.asarray(data_mhd['mb_data']['dens'])
  B1  = np.asarray(data_mhd['mb_data']['bcc1'])
  B2  = np.asarray(data_mhd['mb_data']['bcc2'])
  B3  = np.asarray(data_mhd['mb_data']['bcc3'])
  v1  = np.asarray(data_mhd['mb_data']['velx'])
  v2  = np.asarray(data_mhd['mb_data']['vely'])
  v3  = np.asarray(data_mhd['mb_data']['velz'])

  # MeshBlock statistics.
  NZ = int(np.shape(x3)[1])
  NY = int(np.shape(x2)[1])
  NX = int(np.shape(x1)[1])

  # Mask the blocks at the target resolution.
  mask = np.isclose(dx, target_dx)
  fine_mbs = np.flatnonzero(mask)
  minx, maxx = x1[mask,0].min(), x1[mask,NX-1].max()
  miny, maxy = x2[mask,0].min(), x2[mask,NY-1].max()
  minz, maxz = x3[mask,0].min(), x3[mask,NZ-1].max()

  # Fill the fine arrays.
  fine_points_x = int((maxx - minx) / target_dx) + 1
  fine_points_y = int((maxy - miny) / target_dx) + 1
  fine_points_z = int((maxz - minz) / target_dx) + 1
  finerho = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finevx  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finevy  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finevz  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finebx  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  fineby  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finebz  = np.zeros((fine_points_z, fine_points_y, fine_points_x))
  finex   = np.zeros(fine_points_x)
  finey   = np.zeros(fine_points_y)
  finez   = np.zeros(fine_points_z)

  for m in fine_mbs:
    zind = int((x3[m,0] - minz) / target_dx)
    yind = int((x2[m,0] - miny) / target_dx)
    xind = int((x1[m,0] - minx) / target_dx)
    finerho[zind:zind+NZ, yind:yind+NY, xind:xind+NX] = rho[m,:,:,:]
    finevx[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = v1[m,:,:,:]
    finevy[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = v2[m,:,:,:]
    finevz[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = v3[m,:,:,:]
    finebx[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = B1[m,:,:,:]
    fineby[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = B2[m,:,:,:]
    finebz[zind:zind+NZ, yind:yind+NY, xind:xind+NX]  = B3[m,:,:,:]
    finex[xind:xind+NX] = x1[m,:]
    finey[yind:yind+NY] = x2[m,:]
    finez[zind:zind+NZ] = x3[m,:]

  # Mask the points on the window radius (relative to the remnant center).
  r = np.sqrt((finex[None,None,:] - center[0])**2 +
              (finey[None,:,None] - center[1])**2 +
              (finez[:,None,None] - center[2])**2)
  finevx[r>window_rad] = finevx[r>window_rad] * np.exp(-(r[r>window_rad]))
  finevy[r>window_rad] = finevy[r>window_rad] * np.exp(-(r[r>window_rad]))
  finevz[r>window_rad] = finevz[r>window_rad] * np.exp(-(r[r>window_rad]))
  finebx[r>window_rad] = finebx[r>window_rad] * np.exp(-(r[r>window_rad]))
  fineby[r>window_rad] = fineby[r>window_rad] * np.exp(-(r[r>window_rad]))
  finebz[r>window_rad] = finebz[r>window_rad] * np.exp(-(r[r>window_rad]))

  finekex = np.sqrt(finerho) * finevx
  finekey = np.sqrt(finerho) * finevy
  finekez = np.sqrt(finerho) * finevz

  # Compute the N-dim. discrete Fourier transform
  # on the magnetic field components and the velocity.
  # Use B directly (not B/sqrt(rho)).
  ftbx  = np.fft.fftn(finebx)
  ftby  = np.fft.fftn(fineby)
  ftbz  = np.fft.fftn(finebz)
  ftvx  = np.fft.fftn(finevx)
  ftvy  = np.fft.fftn(finevy)
  ftvz  = np.fft.fftn(finevz)
  ftkex = np.fft.fftn(finekex)
  ftkey = np.fft.fftn(finekey)
  ftkez = np.fft.fftn(finekez)

  # Return the discrete Fourier transform sample
  # frequencies and shift the zero-frequency component
  # to the center of the spectrum. The fine grid isn't necessarily a cube
  # (the finest-level MeshBlocks don't have to span equal extents along
  # x/y/z), so each axis needs its own frequency array.
  ftkz  = np.fft.fftshift(np.fft.fftfreq(finebx.shape[0], d=target_dx))
  ftky  = np.fft.fftshift(np.fft.fftfreq(finebx.shape[1], d=target_dx))
  ftkx  = np.fft.fftshift(np.fft.fftfreq(finebx.shape[2], d=target_dx))
  ftbx  = np.fft.fftshift(ftbx)
  ftby  = np.fft.fftshift(ftby)
  ftbz  = np.fft.fftshift(ftbz)
  ftvx  = np.fft.fftshift(ftvx)
  ftvy  = np.fft.fftshift(ftvy)
  ftvz  = np.fft.fftshift(ftvz)
  ftkex = np.fft.fftshift(ftkex)
  ftkey = np.fft.fftshift(ftkey)
  ftkez = np.fft.fftshift(ftkez)

  # Normalize.
  ftbx  = ftbx / ftbx.size
  ftby  = ftby / ftby.size
  ftbz  = ftbz / ftbz.size
  ftvx  = ftvx / ftvx.size
  ftvy  = ftvy / ftvy.size
  ftvz  = ftvz / ftvz.size
  ftkex = ftkex / ftkex.size
  ftkey = ftkey / ftkey.size
  ftkez = ftkez / ftkez.size

  # Interpolate the FT of magnetic field and velocity onto
  # the sample grid. Grid axes match the array axis order (z, y, x).
  interpftbx  = RegularGridInterpolator((ftkz,ftky,ftkx), ftbx, method='linear')
  interpftby  = RegularGridInterpolator((ftkz,ftky,ftkx), ftby, method='linear')
  interpftbz  = RegularGridInterpolator((ftkz,ftky,ftkx), ftbz, method='linear')
  interpftvx  = RegularGridInterpolator((ftkz,ftky,ftkx), ftvx, method='linear')
  interpftvy  = RegularGridInterpolator((ftkz,ftky,ftkx), ftvy, method='linear')
  interpftvz  = RegularGridInterpolator((ftkz,ftky,ftkx), ftvz, method='linear')
  interpftkex = RegularGridInterpolator((ftkz,ftky,ftkx), ftkex, method='linear')
  interpftkey = RegularGridInterpolator((ftkz,ftky,ftkx), ftkey, method='linear')
  interpftkez = RegularGridInterpolator((ftkz,ftky,ftkx), ftkez, method='linear')

  # Create a spherical grid. Cap the sampled radius to the smallest of the
  # three axes' Nyquist limits, so the spherical samples below never fall
  # outside any interpolator's grid.
  maxrad = min(ftkx[-1], ftky[-1], ftkz[-1])
  nrad   = int(min(finebx.shape) / 2 + 1)
  ntheta = 30
  nphi = 60

  kr_mhd    = np.linspace(0, maxrad, nrad)
  ktheta    = np.linspace(0, np.pi, ntheta)
  kphi      = np.linspace(0, 2.0 * np.pi, nphi)
  sinktheta = np.sin(ktheta)
  cosktheta = np.cos(ktheta)
  sinkphi   = np.sin(kphi)
  coskphi   = np.cos(kphi)
  dktheta   = np.pi / ntheta
  dkphi     = 2.0 * np.pi / nphi

  powerb_mhd  = np.zeros(len(kr_mhd))
  powerv_mhd  = np.zeros(len(kr_mhd))
  powerke_mhd = np.zeros(len(kr_mhd))

  kz = np.zeros((nphi, ntheta, nrad))
  ky = np.zeros((nphi, ntheta, nrad))
  kx = np.zeros((nphi, ntheta, nrad))

  kx = kr_mhd[None,None,:] * sinktheta[None,:,None] * coskphi[:,None,None]
  ky = kr_mhd[None,None,:] * sinktheta[None,:,None] * sinkphi[:,None,None]
  kz = kr_mhd[None,None,:] * cosktheta[None,:,None]

  bxsph  = interpftbx((kz,ky,kx) )
  bysph  = interpftby((kz,ky,kx) )
  bzsph  = interpftbz((kz,ky,kx) )
  vxsph  = interpftvx((kz,ky,kx) )
  vysph  = interpftvy((kz,ky,kx) )
  vzsph  = interpftvz((kz,ky,kx) )
  kexsph = interpftkex((kz,ky,kx) )
  keysph = interpftkey((kz,ky,kx) )
  kezsph = interpftkez((kz,ky,kx) )

  powerb_mhd  = np.sum(np.sum((np.absolute(bxsph)**2 + np.absolute(bysph)**2 + np.absolute(bzsph)**2)
                              * dktheta * dkphi * sinktheta[None,:,None] * kr_mhd[None,None,:]**2, axis=0), axis=0)
  powerv_mhd  = np.sum(np.sum((np.absolute(vxsph)**2 + np.absolute(vysph)**2 + np.absolute(vzsph)**2)
                              * dktheta * dkphi * sinktheta[None,:,None] * kr_mhd[None,None,:]**2, axis=0), axis=0)
  powerke_mhd = np.sum(np.sum((np.absolute(kexsph)**2 + np.absolute(keysph)**2 + np.absolute(kezsph)**2)
                              * dktheta * dkphi * sinktheta[None,:,None] * kr_mhd[None,None,:]**2, axis=0), axis=0)

  del data_mhd # Free up memory.

  # Save this snapshot's spectra to its own file.
  out_path = os.path.join(outdir, "spectra", f"spectrum_{snapshot}.txt")
  np.savetxt(out_path, np.c_[kr_mhd, powerv_mhd, powerb_mhd, powerke_mhd],
             header=f"Time = {time}\n0:k    1:P_v    2:P_B    3:P_KE",
             comments="# ")


def analyze(args):
  """Run the spectra analysis over every snapshot in `bindir`.

  Parameters (from args; in detail)
  ---------------------------------
  args.bindir (str): path to the 3D binary files. IMPORTANT: files need
                     to hold the signature *._3D.*.bin. The output needed
                     at each output time is `*.mhd_w_bcc_3D.*.bin`.
  args.outdir (str): output directory where the snapshots are stored.
  args.drop_first_bins (int): how many bins to drop at the start (during inspiral).
  args.target_dx (float): target resolution on the finest MeshBlock.
  args.win_radius (float): windowing radius.
  args.tracker (str): path to the tracker file used to center the window
                      on the remnant as it drifts across the domain.
  args.center (list): fixed center of the remnant, used if no tracker is given.
  args.n_workers (int): number of worker processes per snapshot loop.
  """
  # Find the files with the signature.
  files = find_snapshots("mhd_w_bcc", args.bindir)

  # Drop snapshots at the beginning.
  if args.drop_first_bins is not None:
    files = [
      f for f in sorted(files)
      if int(re.search(r"\.(\d+)\.bin$", f).group(1)) >= args.drop_first_bins
    ]

  # Load the tracker file, if given.
  tracker = load_tracker(args.tracker) if args.tracker is not None else None

  # Prepare the worker arguments.
  n_workers = args.n_workers if args.n_workers is not None else min(8, cpu_count())
  worker_args = [(files[i], args.target_dx, args.win_radius, args.outdir, args.center)
                 for i in range(len(files))]
  print(f"$ Processing {len(worker_args)} snapshots with n_worker={n_workers}...")

  # Make directories.
  os.makedirs(os.path.join(args.outdir, "spectra"), exist_ok=True)

  # Parallel snapshot loop.
  if n_workers > 1:
    with Pool(n_workers,
              initializer=_init_spectra_worker,
              initargs=(tracker,)) as pool:
      for _ in tqdm(
        pool.imap_unordered(
          _process_snapshot_spectra, worker_args,
          chunksize=max(1, len(worker_args) // (n_workers * 4))),
          total=len(worker_args),
          desc="Progress"):
        pass
  else:
    _init_spectra_worker(tracker)
    for a in tqdm(worker_args, desc="Progress"):
      _process_snapshot_spectra(a)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main(argv=None):
  ap = argparse.ArgumentParser(
      description="Post-merger disk diagnostics for AthenaK data.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  ap.add_argument("--bindir", required=True,
                  help="Directory holding the 3D bin files.")
  ap.add_argument("--outdir", required=True,
                    help="Output directory path.")
  ap.add_argument("--drop-first-bins", default=None, type=int,
                    help="Drop that many .bin files at the start (inspiral).")
  ap.add_argument("--target-dx", required=True, type=float,
                  help="Resolution on the finest MeshBlock.")
  ap.add_argument("--win-radius", type=float,
                  help="Windowing radius.")
  ap.add_argument("--tracker", default=None,
                  help="Path to the tracker file used to center the window on the remnant.")
  ap.add_argument("--center", type=float, nargs=3, default=None,
                  metavar=("X", "Y", "Z"),
                  help="Fixed center of the remnant, used if no tracker is given.")
  ap.add_argument("--n-workers", type=int, default=None,
                  help="Worker processes for the snapshot loop. "
                        "Default: min(8, cpu_count()).")
  args = ap.parse_args()

  analyze(args)


if __name__ == "__main__":
  main()
