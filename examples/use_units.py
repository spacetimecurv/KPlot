#########################################################################
# File: use_units.py                                                    #
# Description: Use unit conversion between unit systems.                #
#########################################################################

# Import the Units class.
from kplot import Units, CGS, MKS, GEOMETRIC_SOLAR

# Define the units systems.
cgs = CGS
mks = MKS
geo = GEOMETRIC_SOLAR

# Print the unit systems.
cgs.print_units()
mks.print_units()
geo.print_units()

# Convert quantities.
energy_geo_to_cgs = geo.energy_conversion(cgs) # Geometric -> CGS
print(f"Energy conversion geometric to cgs units: {energy_geo_to_cgs:.5e} [ergs]")
