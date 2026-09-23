#########################################################################
# File: units.py                                                        #
# Description: Unit conversions for AthenaK data.                       #
#########################################################################

# Import third-party libraries.
from matplotlib.colors import LogNorm, Normalize

# Define the Units class.
class Units:
  """
  A class defining different units systems and conversions.
  """

  def __init__(self,
               unit_system,
               c, G, kb, Msun, MeV,
               length, time, density, mass,
               energy, pressure, temperature, chemical_potential):
    """
    Instantiate the Units class, by defining constants/derived quantities.

    Parameters:
    unit_system (str): The used unit system.
    c (float): Speed of light.
    G (float): Gravitational constant.
    kb (float): Boltzmann constant.
    Msun (float): Solar mass.
    Mev (float): Temperature.
    length (float): Length scale.
    time (float): Time scale.
    density (float): Density scale.
    mass (float): Mass scale.
    energy (float): Energy scale.
    pressure (float): Pressure scale.
    temperature (float): Temperature scale.
    chemical_potential (float): Chemical potential.
    """
    self.unit_system = unit_system

    self.c = c
    self.G = G
    self.kb = kb
    self.Msun = Msun
    self.MeV = MeV

    self.length = length
    self.time = time
    self.density = density
    self.mass = mass
    self.energy = energy
    self.pressure = pressure
    self.temperature = temperature
    self.chemical_potential = chemical_potential

  # Conversion between length of one Unit system with another.
  def length_conversion(self, b):
    """
    Converts length between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.length / self.length

  # Conversion between time of one Unit system with another.
  def time_conversion(self, b):
    """
    Converts time between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.time / self.time

  # Conversion between velocity of one Unit system with another.
  def velocity_conversion(self, b):
    """
    Converts velocity between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return (b.length / self.length) * (self.time / b.time)

  # Conversion between density of one Unit system with another.
  def density_conversion(self, b):
    """
    Converts density between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.density / self.density

  # Conversion between mass of one Unit system with another.
  def mass_conversion(self, b):
    """
    Converts mass between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.mass / self.mass

  # Conversion between mass density of one Unit system with another.
  def mass_density_conversion(self, b):
    """
    Converts mass density between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return (b.density / self.density) * (b.mass / self.mass)

  # Conversion between energy of one Unit system with another.
  def energy_conversion(self, b):
    """
    Converts energy between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.energy / self.energy

  # Conversion between energy density of one Unit system with another.
  def energy_density_conversion(self, b):
    """
    Converts energy density between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return (b.density / self.density) * (b.energy / self.energy)

  # Conversion between pressure of one Unit system with another.
  def pressure_conversion(self, b):
    """
    Converts pressure between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.pressure / self.pressure

  # Conversion between temperature of one Unit system with another.
  def temperature_conversion(self, b):
    """
    Converts temperature between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.temperature / self.temperature

  # Conversion between chemical potential of one Unit system with another.
  def chemical_potential_conversion(self, b):
    """
    Converts chemical potential between unit systems.

    Parameters:
    b (Units): Unit system to be converted into.
    """
    return b.chemical_potential / self.chemical_potential

  # Function to print the current unit system.
  def print_units(self):
    """
    Prints the current unit system.
    """
    print(f"=============== {self.unit_system} ===============")
    print(f"$ Speed of light: {self.c}")
    print(f"$ Gravitational constant: {self.G}")
    print(f"$ Boltzmann constant: {self.kb}")
    print(f"$ Solar mass: {self.Msun}")
    print(f"$ Temperature: {self.MeV}\n")
    print(f"$ Length scale: {self.length}")
    print(f"$ Time scale: {self.time}")
    print(f"$ Density scale: {self.density}")
    print(f"$ Mass scale: {self.mass}")
    print(f"$ Energy scale: {self.energy}")
    print(f"$ Pressure scale: {self.pressure}")
    print(f"$ Temperature scale: {self.temperature}")
    print(f"$ Chemical potential scale: {self.chemical_potential}")
    print("Done.\n")

  # CGS unit system based on CODATA values (2014).
  @staticmethod
  def CGS():
    """
    Defines the CGS unit system.

    Returns:
    Units instance set with CGS properties.
    """
    return Units(
      "CGS",
      2.99792458e10,   # cm / s
      6.67408e-8,      # cm^3 g^-1 s^-2
      1.38064852e-16,  # erg K^-1
      1.98848e33,      # g
      1.6021766208e-6, # erg
      1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0
    )

  # Gemetric solar unit system.
  @staticmethod
  def GeometricSolar():
    """
    Defines the geometric solar unit system.

    Returns:
    Units instance set with geometric solar properties.
    """
    cgs = Units.CGS()

    return Units(
      "GEOMETRIC_SOLAR",
      1.0,
      1.0,
      1.0,
      1.0,
      cgs.MeV / (cgs.c * cgs.c),

      (cgs.c * cgs.c) / (cgs.G * cgs.Msun),
      (cgs.c**3) / (cgs.G * cgs.Msun),
      ((cgs.G * cgs.Msun) / (cgs.c * cgs.c))**3,
      1.0 / cgs.Msun,
      1.0 / (cgs.Msun * cgs.c * cgs.c),
      ((cgs.G / (cgs.c * cgs.c))**3) * ((cgs.Msun / cgs.c)**2),
      cgs.kb / cgs.MeV,
      cgs.kb / cgs.MeV
    )

  # MKS unit system.
  @staticmethod
  def MKS():
    """
    Defines the MKS unit system.

    Returns:
    Units instance set with MKS properties.
    """
    cgs = Units.CGS()
    return Units(
      "MKS",
      cgs.c / 1e2,
      cgs.G / 1e3,
      cgs.kb / 1e7,
      cgs.Msun / 1e3,
      cgs.MeV / 1e7,
      1e-2,
      1.0,
      1e6,
      1e-3,
      1e-7,
      0.1,
      1.0,
      1e-7
    )

# Global singeltons.
CGS = Units.CGS()
MKS = Units.MKS()
GEOMETRIC_SOLAR = Units.GeometricSolar()

# ****************** ATHENAK UNIT CONVENTIONS **********************
class AthenaK_Units:
  """
  A class defining AthenaK units.
  """

  def __init__(self):
    """
    Instantiate the AthenaK units class.
    """
    self.cgs = Units.CGS()

    # GeometricKilometer (GK) unit scales — G = c = 1, length unit = 1 km
    self.GK_press  = self.cgs.G / self.cgs.c**4 * 1e10 # 1 erg/cm^3  expressed in km^-2
    self.GK_volume = 1e-15                             # 1 cm^3      expressed in km^3
    self.GK_energy = self.cgs.G / self.cgs.c**4 * 1e-5                   # 1 erg       expressed in km

    # NGS (nm, g, s, MeV energy)
    self.NGS_volume = 1e21               # 1 cm^3  in nm^3   (1 cm = 1e7 nm)
    self.NGS_numden = 1e-21              # 1 cm^-3 in nm^-3

    # Nuclear (fm, MeV)
    self.NUC_volume = 1e39               # 1 cm^3  in fm^3   (1 cm = 1e13 fm)

    # Derived conversion constants.
    # Energy density: code (km^-2) -> MeV/nm^3
    self.GK2NGS_ENEDENS = (self.GK_volume / self.NGS_volume) \
                           * (1.0 / self.cgs.MeV / self.GK_energy)

    # Energy density: code (km^-2) -> erg/cm^3
    self.CODE2CGS_ENEDENS = 1.0 / self.GK_press

    # Radiation number density: code (fm^-3) -> nm^-3
    self.CODE2NGS_NUMDENS = 1e18  # 1 fm^-3 = 1e18 nm^-3

    # Radiation number density: code (fm^-3) -> cm^-3
    self.CODE2CGS_NUMDENS = 1e39  # 1 fm^-3 = 1e39 cm^-3

    # Opacity: code (km^-1) -> nm^-1
    self.CODE2NGS_OPACITY = 1e-12 # 1 km^-1 = 1e-12 nm^-1

    # Opacity: code (km^-1) -> cm^-1
    self.CODE2CGS_OPACITY = 1e-5  # 1 km^-1 = 1e-5  cm^-1

    # Number emissivity eta_0: code -> nm^-3 s^-1
    self.c_km        = self.cgs.c * 1e-5                 # c in km/s  ~ 2.998e5 km/s
    self.UNIT_ND_DOT = self.CODE2NGS_NUMDENS * self.c_km # ~ 2.998e23

    # Number emissivity eta_0: code -> cm^-3 s^-1
    self.CODE2CGS_ND_DOT = self.UNIT_ND_DOT * self.NGS_numden # / 1e21

    # Energy emissivity eta_1: code -> MeV nm^-3 s^-1
    self.UNIT_ED_DOT = self.GK2NGS_ENEDENS * self.c_km # ~ 2.265e29

    # Energy emissivity eta_1: code -> erg cm^-3 s^-1
    self.CODE2CGS_ED_DOT = self.UNIT_ED_DOT * self.cgs.MeV * self.NGS_numden

    # Rest-mass density: code (G=c=M_sun=1) -> g/cm^3
    self.unit_len_cgs  = self.cgs.G * self.cgs.Msun / self.cgs.c**2 # ~ 1.477e6 cm
    self.CODE2CGS_DENS = self.cgs.Msun / self.unit_len_cgs**3       # ~ 6.178e17

    # Average neutrino energy E/N: code ratio (km^-2)/(fm^-3) -> MeV
    self.CODE2MEV_AVGENE = self.GK2NGS_ENEDENS / self.CODE2NGS_NUMDENS  # ~ 7.556e5

  # Explicit conversion functions.
  def conv_dens(self, val, units='cgs'):
    """Rest-mass density (code G=c=Msun=1 units) -> target units."""
    if units == 'code': return val
    if units == 'cgs' : return val * self.CODE2CGS_DENS
    raise ValueError(f"density: unsupported units {units!r} (use 'code' or 'cgs')")


  def conv_enedens(self, val, units='cgs'):
    """Radiation energy density E (code km^-2) -> target units."""
    if units == 'code': return val
    if units == 'ngs' : return val * self.GK2NGS_ENEDENS
    if units == 'cgs' : return val * self.CODE2CGS_ENEDENS
    raise ValueError(f"enedens: unsupported units {units!r}")


  def conv_numdens(self, val, units='cgs'):
    """Radiation/baryon number density N (code fm^-3) -> target units."""
    if units == 'code': return val
    if units == 'ngs' : return val * self.CODE2NGS_NUMDENS
    if units == 'cgs' : return val * self.CODE2CGS_NUMDENS
    raise ValueError(f"numdens: unsupported units {units!r}")


  def conv_opacity(self, val, units='cgs'):
    """Absorption/scattering opacity kappa (code km^-1) -> target units."""
    if units == 'code': return val
    if units == 'ngs' : return val * self.CODE2NGS_OPACITY
    if units == 'cgs' : return val * self.CODE2CGS_OPACITY
    raise ValueError(f"opacity: unsupported units {units!r}")


  def conv_emissivity_N(self, val, units='cgs'):
    """Number emissivity eta_0 (code) -> target units."""
    if units == 'code': return val
    if units == 'ngs' : return val * self.UNIT_ND_DOT
    if units == 'cgs' : return val * self.CODE2CGS_ND_DOT
    raise ValueError(f"emissivity_N: unsupported units {units!r}")


  def conv_emissivity_E(self, val, units='cgs'):
    """Energy emissivity eta_1 (code) -> target units."""
    if units == 'code': return val
    if units == 'ngs' : return val * self.UNIT_ED_DOT
    if units == 'cgs' : return val * self.CODE2CGS_ED_DOT
    raise ValueError(f"emissivity_E: unsupported units {units!r}")


  def conv_avg_energy(self, val, units='MeV'):
    """Average neutrino energy E/N (code) -> target units."""
    if units == 'code': return val
    if units in ('ngs', 'MeV'): return val * self.CODE2MEV_AVGENE
    if units == 'cgs':          return val * self.CODE2MEV_AVGENE * self.cgs.MeV
    raise ValueError(f"avg_energy: unsupported units {units!r}")
