#########################################################################
# File: units.py                                                        #
# Description: Unit conversions for AthenaK data.                       #
#########################################################################

# Import necessary standard libraries.


# Import necessary third-party libraries.


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
