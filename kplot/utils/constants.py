"""Code-unit (G = c = M_sun = 1) conversions, derived from `kplot.system.units`."""

from kplot.system.units import CGS, GEOMETRIC_SOLAR

LENGTH_CM  = 1.0 / GEOMETRIC_SOLAR.length  # 1 code length in cm
MSUN_TO_S  = 1.0 / GEOMETRIC_SOLAR.time
MSUN_TO_MS = MSUN_TO_S * 1e3

# Code power (M_sun c^2 per code time) in erg/s, i.e. c^5/G.
CODE_POWER_CGS = CGS.c**5 / CGS.G
