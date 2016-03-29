from __future__ import print_function, division, absolute_import
import numpy as np


def refraction(wavelength, zenith_angle, atmospheric_pressure, temperature, humidity=10,
               latitude=-30.244639, altitude=2663.):
    """Calculate overall refraction under atmospherica and observing conditions.
    input pressure is in atmospheres
    wavelength and zenith_angle may be supplied as numpy vectors
    wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    temperature is in Celcius (valid for -20 < T < 50)
    humidity is in percent (0-100)
    zenith angle is in degrees. Equations are valid for zenith_angle < 75
    latitude is in degrees
    altitude is in meters
    """
    temperature_Kelvin = temperature + 273.15
    water_vapor_pressure = humidity_to_pressure(humidity=humidity, temperature=temperature)

    atm_to_millibar = 760. * 1.333224
    dry_pressure = atm_to_millibar * atmospheric_pressure - water_vapor_pressure

    reduced_n = n_delta(wavelength, dry_pressure, water_vapor_pressure, temperature_Kelvin) * 1E-8

    atmos_scaleheight_ratio = 4.5908E-6 * temperature_Kelvin

    # Account for oblate Earth
    relative_gravity = (1. + 0.005302 * np.sin(np.radians(latitude))**2. +
                        -0.00000583 * np.sin(np.radians(2. * latitude))**2. - 0.000000315 * altitude)

    tanZ = np.tan(np.radians(zenith_angle))

    atmos_term_1 = reduced_n * relative_gravity * (1. - atmos_scaleheight_ratio)
    atmos_term_2 = reduced_n * relative_gravity * (atmos_scaleheight_ratio - reduced_n / 2.)
    if (type(zenith_angle) == np.ndarray) and (type(wavelength) == np.ndarray):
        atmos_term_1 = np.matrix(atmos_term_1)
        atmos_term_2 = np.matrix(atmos_term_2)
        tanZ = np.matrix(tanZ)
        result = np.array(atmos_term_1.T * tanZ + atmos_term_2.T * np.power(tanZ, 3.))
    else:
        result = atmos_term_1 * tanZ + atmos_term_2 * np.power(tanZ, 3.)
    return np.degrees(result)


def diff_refraction(wavelength=None, wavelength_ref=None, zenith_angle=None, atmospheric_pressure=1.,
                    humidity=10., temperature=20., latitude=-30.244639, altitude=2663., **kwargs):
    """Calculate the differential refraction between two wavelengths."""
    refraction_start = refraction(wavelength, zenith_angle, atmospheric_pressure, temperature,
                                  humidity=humidity, latitude=latitude, altitude=altitude)
    refraction_end = refraction(wavelength_ref, zenith_angle, atmospheric_pressure, temperature,
                                humidity=humidity, latitude=latitude, altitude=altitude)
    return refraction_start - refraction_end


def n_delta(wavelength, dry_pressure, water_vapor_pressure=0.0, temperature=300.0):
    """Calculate difference of refractive index of air from 1, multiplied by 1E8.

    temperature is in units of Kelvin
    pressures are in units of millibar
    want wave number in units 1/micron
    wavelength is input in nm
    """
    wave_num = 1E3 / wavelength

    dry_air_term = (2371.34 + (683939.7 / (130. - np.power(wave_num, 2.))) +
                    (4547.3 / (38.9 - np.power(wave_num, 2.))))

    wet_air_term = (6487.31 + 58.058 * np.power(wave_num, 2.) +
                    -0.71150 * np.power(wave_num, 4.) + 0.08851 * np.power(wave_num, 6.))

    return (dry_air_term * density_factor_dry(dry_pressure, temperature) +
            wet_air_term * density_factor_water(water_vapor_pressure, temperature))


def density_factor_dry(dry_pressure, temperature):
    """Calculate dry air pressure term to refractive index calculation."""
    eqn_1 = (1. + dry_pressure * (57.90E-8 - (9.3250E-4 / temperature)
             + (0.25844 / np.power(temperature, 2.))))

    return eqn_1 * dry_pressure / temperature


def density_factor_water(water_vapor_pressure, temperature):
    """Calculate water vapor pressure term to refractive index calculation."""
    eqn_1 = ((-2.37321E-3 + (2.23366 / temperature) - (710.792 / np.power(temperature, 2.)) +
             (7.75141E-4 / np.power(temperature, 3.))))
    eqn_2 = 1. + water_vapor_pressure * (1. + 3.7E-4 * water_vapor_pressure) * eqn_1

    return eqn_2 * water_vapor_pressure / temperature


def humidity_to_pressure(humidity=20., temperature=15.):
    """Simple function that converts humidity and temperature to water vapor pressure."""
    pascals_to_mbar = 60. * 1.333224 / 101325.0
    temperature_Kelvin = temperature + 273.15
    saturation_pressure = (pascals_to_mbar * np.exp(77.3450 + 0.0057 * temperature_Kelvin +
                           -7235.0 / temperature_Kelvin) / np.power(temperature_Kelvin, 8.2))
    return (humidity / 100.0) * saturation_pressure
