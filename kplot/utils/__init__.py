"""Helpers shared by `kplot.system`, `kplot.sphere` and `kplot.volume`.

- ``constants``  - physical constants and code-unit conversions
- ``grmhd``      - 3+1 metric algebra, fluid four-velocity and mag. fields
- ``eos``        - PyCompOSE EOS tables
- ``parallel``   - per-snapshot process pool
- ``sph``        - locating and reading sph surface dumps
- ``quadrature`` - surface and time integrals
- ``surface``    - generic per-snapshot function for sph analyses
- ``bin3d``      - 3D binary dumps, tracker and horizon files
- ``cli``        - shared command-line options
- ``plotting``   - time axis and figure saving
"""
