"""Function to generate simulated catalogs with reproduceable source spectra to feed into fast_dft."""
from __future__ import print_function, division, absolute_import
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
from scipy import constants
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.sims.photUtils import Bandpass, matchStar  # , Sed, PhotometricParameters
from lsst.utils import getPackageDir
# import galsim
# from lsst.sims.photUtils import matchStar
# import lsst.afw.image as afwImage
from calc_refractive_index import diff_refraction
from fast_dft import fast_dft
import time
photons_per_adu = 1e4  # used only to approximate the effect of photon shot noise, if photon_noise=True


def cat_image(catalog=None, bbox=None, name=None, psf=None, pixel_scale=None, pad_image=1.5,
              sky_noise=0.0, instrument_noise=0.0, photon_noise=False,
              dcr_flag=False, band_name='g', sed_list=None,
              astrometric_error=None, edge_dist=None, **kwargs):
    """!Wrapper that takes a catalog of stars and simulates an image in units of Janskys."""
    """
    if psf is None:
        psf = galsim.Kolmogorov(fwhm=1)
    """
    # I think most PSF classes have a getFWHM method. The math converts to a sigma for a gaussian.
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2. * np.log(2)))
    if pixel_scale is None:
        pixel_scale = psf.getFWHM() * fwhm_to_sigma
    if edge_dist is None:
        if pad_image > 1:
            edge_dist = 0
        else:
            edge_dist = 5 * psf.getFWHM() * fwhm_to_sigma / pixel_scale
    kernel_radius = np.ceil(5 * psf.getFWHM() * fwhm_to_sigma / pixel_scale)
    # print("Kernel radius used: ", kernel_radius)
    if catalog is None:
        catalog = cat_sim(bbox=bbox, name=name, edge_distance=edge_dist, **kwargs)
    schema = catalog.getSchema()
    n_star = len(catalog)
    bandpass = load_bandpass(band_name=band_name, **kwargs)
    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    if name is None:
        # If no name is supplied, find the first entry in the schema in the format *_flux
        schema_entry = schema.extract("*_flux", ordered='true')
        fluxName = schema_entry.iterkeys().next()
    else:
        fluxName = name + '_flux'

    if sed_list is None:
        # Load in model SEDs
        matchStarObj = matchStar()
        sed_list = matchStarObj.loadKuruczSEDs()

    fluxKey = schema.find(fluxName).key
    temperatureKey = schema.find("temperature").key
    metalKey = schema.find("metallicity").key
    gravityKey = schema.find("gravity").key
    x0, y0 = bbox.getBegin()
    # if catalog.isContiguous()
    flux = catalog[fluxKey] / psf.getFlux()
    temperatures = catalog[temperatureKey]
    metallicities = catalog[metalKey]
    gravities = catalog[gravityKey]
    flux_arr = np.zeros((n_star, bandpass_nstep(bandpass)))

    for _i in range(n_star):
        f_star = flux[_i]
        t_star = temperatures[_i]
        z_star = metallicities[_i]
        g_star = gravities[_i]
        star_spectrum = star_gen(sed_list=sed_list, temperature=t_star, flux=f_star, bandpass=bandpass,
                                 metallicity=z_star, surface_gravity=g_star)
        flux_arr[_i, :] = np.array([flux_val for flux_val in star_spectrum])
    flux_tot = np.sum(flux_arr, axis=1)
    if n_star > 3:
        cat_sigma = np.std(flux_tot[flux_tot - np.median(flux_tot) < 3.0 * np.std(flux_tot)])
        i_bright = (np.where(flux_tot - np.median(flux_tot) > 3.0 * cat_sigma))[0]
        n_bright = len(i_bright)
        i_faint = (np.where(flux_tot - np.median(flux_tot) <= 3.0 * cat_sigma))[0]
        n_faint = len(i_faint)
    else:
        i_bright = np.arange(n_star)
        i_faint = np.arange(0)
        n_bright = n_star
        n_faint = 0
    if not dcr_flag:
        flux_arr = flux_tot
        flux_bright = flux_arr[i_bright]
        flux_arr = flux_arr[i_faint]
    else:
        flux_bright = flux_arr[i_bright, :]
        flux_arr = flux_arr[i_faint, :]

    xv = catalog.getX() - x0
    yv = catalog.getY() - y0

    return_image = np.zeros((y_size, x_size))
    if dcr_flag:
        if n_faint > 0:
            return_image += convolve_dcr_image(flux_arr, xv[i_faint], yv[i_faint],
                                               bandpass=bandpass, x_size=x_size, y_size=y_size,
                                               kernel_radius=kernel_radius,
                                               psf=psf, pad_image=pad_image, pixel_scale=pixel_scale,
                                               photon_noise=photon_noise, sky_noise=sky_noise, **kwargs)
        if n_bright > 0:
            return_image += convolve_dcr_image(flux_bright, xv[i_bright], yv[i_bright],
                                               bandpass=bandpass, x_size=x_size, y_size=y_size,
                                               kernel_radius=x_size, oversample_image=2.0,
                                               psf=psf, pad_image=pad_image, pixel_scale=pixel_scale,
                                               photon_noise=photon_noise, sky_noise=0.0, **kwargs)

    else:
        if n_faint > 0:
            return_image += convolve_image(flux_arr, xv[i_faint], yv[i_faint],
                                           x_size=x_size, y_size=y_size, kernel_radius=kernel_radius,
                                           psf=psf, pad_image=pad_image, pixel_scale=pixel_scale,
                                           photon_noise=photon_noise, sky_noise=sky_noise, **kwargs)
        if n_bright > 0:
            return_image += convolve_image(flux_bright, xv[i_bright], yv[i_bright],
                                           x_size=x_size, y_size=y_size,
                                           kernel_radius=x_size, oversample_image=2.0,
                                           psf=psf, pad_image=pad_image, pixel_scale=pixel_scale,
                                           photon_noise=photon_noise, sky_noise=0.0, **kwargs)
    if instrument_noise > 0:
        return_image += np.random.normal(scale=instrument_noise, size=(y_size, x_size))
    return(return_image)


def convolve_dcr_image(flux_arr, x_loc, y_loc, bandpass=None, x_size=None, y_size=None,
                       psf=None, pad_image=1.5, pixel_scale=None, kernel_radius=None,
                       oversample_image=1, photon_noise=False, sky_noise=0.0, verbose=True, **kwargs):
    """Wrapper to call fast_dft with multiple DCR planes."""
    x_size_use = int(x_size * pad_image)
    y_size_use = int(y_size * pad_image)
    oversample_image = int(oversample_image)
    pixel_scale_use = pixel_scale / oversample_image
    x0 = oversample_image * ((x_size_use - x_size) // 2)
    x1 = x0 + x_size * oversample_image
    y0 = oversample_image * ((y_size_use - y_size) // 2)
    y1 = y0 + y_size * oversample_image
    x_loc_use = x_loc * oversample_image + x0
    y_loc_use = y_loc * oversample_image + y0
    x_size_use *= oversample_image
    y_size_use *= oversample_image
    timing_model = -time.time()
    source_image = fast_dft(flux_arr, x_loc_use, y_loc_use, x_size=x_size_use, y_size=y_size_use,
                            kernel_radius=kernel_radius, **kwargs)
    timing_model += time.time()
    n_star = len(x_loc)
    if oversample_image > 1:
        bright_star = "bright "
    else:
        bright_star = ""
    if verbose:
        if n_star == 1:
            print("Time to model %i %sstar: [%0.3fs]"
                  % (n_star, bright_star, timing_model))
        else:
            print("Time to model %i %sstars: [%0.3fs | %0.5fs per star]"
                  % (n_star, bright_star, timing_model, timing_model / n_star))
    # The images are purely real, so we can save time by using the real FFT,
    # which uses only half of the complex plane
    convol = np.zeros((y_size_use, x_size_use // 2 + 1), dtype='complex64')
    dcr_gen = dcr_generator(bandpass, pixel_scale=pixel_scale_use, **kwargs)
    timing_fft = -time.time()
    for _i, offset in enumerate(dcr_gen):
        source_image_use = source_image[_i]

        psf_image = psf.drawImage(scale=pixel_scale_use, method='fft', offset=offset,
                                  nx=x_size_use, ny=y_size_use, use_true_center=False)
        if photon_noise:
            base_noise = np.random.normal(scale=1.0, size=(y_size_use, x_size_use))
            base_noise *= np.sqrt(np.abs(source_image_use) / photons_per_adu)
            source_image_use += base_noise
        if sky_noise > 0:
            source_image_use += (np.random.normal(scale=sky_noise, size=(y_size_use, x_size_use))
                                 / np.sqrt(bandpass_nstep(bandpass)))
        convol += rfft2(source_image_use) * rfft2(psf_image.array)
    return_image = np.real(fftshift(irfft2(convol)))
    timing_fft += time.time()
    if verbose:
        print("FFT timing for %i DCR planes: [%0.3fs | %0.3fs per plane]"
              % (_i, timing_fft, timing_fft / _i))
    return(return_image[y0:y1:oversample_image, x0:x1:oversample_image] * oversample_image**2)


def convolve_image(flux_arr, x_loc, y_loc, x_size=None, y_size=None,
                   psf=None, pad_image=1.5, pixel_scale=None, kernel_radius=None,
                   oversample_image=1, photon_noise=False, sky_noise=0.0, **kwargs):
    """Wrapper to call fast_dft with no DCR planes."""
    x_size_use = int(x_size * pad_image)
    y_size_use = int(y_size * pad_image)
    oversample_image = int(oversample_image)
    pixel_scale_use = pixel_scale / oversample_image
    x0 = oversample_image * ((x_size_use - x_size) // 2)
    x1 = x0 + x_size * oversample_image
    y0 = oversample_image * ((y_size_use - y_size) // 2)
    y1 = y0 + y_size * oversample_image
    x_loc_use = x_loc * oversample_image + x0
    y_loc_use = y_loc * oversample_image + y0
    x_size_use *= oversample_image
    y_size_use *= oversample_image
    timing_model = -time.time()
    source_image = fast_dft(flux_arr, x_loc_use, y_loc_use, x_size=x_size_use, y_size=y_size_use,
                            kernel_radius=kernel_radius, **kwargs)
    timing_model += time.time()
    n_star = len(x_loc)
    if oversample_image > 1:
        bright_star = "bright "
    else:
        bright_star = ""
    if n_star == 1:
        print("Time to model %i %sstar: [%0.3fs]" % (n_star, bright_star, timing_model))
    else:
        print("Time to model %i %sstars: [%0.3fs | %0.5fs per star]"
              % (n_star, bright_star, timing_model, timing_model / n_star))
    psf_image = psf.drawImage(scale=pixel_scale_use, method='fft', offset=[0, 0],
                              nx=x_size_use, ny=y_size_use, use_true_center=False)
    if photon_noise:
        base_noise = np.random.normal(scale=1.0, size=(y_size_use, x_size_use))
        base_noise *= np.sqrt(np.abs(source_image) / photons_per_adu)
        source_image += base_noise
    if sky_noise > 0:
        source_image += np.random.normal(scale=sky_noise, size=(y_size_use, x_size_use))
    timing_fft = -time.time()
    convol = rfft2(source_image) * rfft2(psf_image.array)
    return_image = np.real(fftshift(irfft2(convol)))
    timing_fft += time.time()
    print("FFT timing (single plane): [%0.3fs]" % (timing_fft))
    return(return_image[y0:y1:oversample_image, x0:x1:oversample_image] * oversample_image**2)


def bandpass_nstep(bandpass):
    """Simple function to pre-compute the number of bins to use for a given bandpass."""
    return(int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step)))


def dcr_generator(bandpass, pixel_scale=None, elevation=None, azimuth=None, **kwargs):
    """!Call the functions that compute Differential Chromatic Refraction (relative to mid-band)."""
    if elevation is None:
        elevation = 50.0
    if azimuth is None:
        azimuth = 0.0
    zenith_angle = 90.0 - elevation
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in wavelength_iterator(bandpass, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = diff_refraction(wavelength=wavelength, wavelength_ref=wavelength_midpoint,
                                      zenith_angle=zenith_angle, **kwargs)
        refract_amp *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
        dx = refract_amp * np.sin(np.radians(azimuth))
        dy = refract_amp * np.cos(np.radians(azimuth))
        yield((dx, dy))


def cat_sim(bbox=None, seed=None, n_star=None, n_galaxy=None, edge_distance=10, name=None, **kwargs):
    """Wrapper function that generates a semi-realistic catalog of stars."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    if name is None:
        name = "sim"
    fluxName = name + "_flux"
    flagName = name + "_flag"
    fluxSigmaName = name + "_fluxSigma"
    schema.addField(fluxName, type="D")
    schema.addField(fluxSigmaName, type="D")
    schema.addField(flagName, type="D")
    schema.addField(name + "_Centroid_x", type="D")
    schema.addField(name + "_Centroid_y", type="D")
    schema.addField("temperature", type="D")
    schema.addField("spectral_id", type="D")
    schema.addField("metallicity", type="D")
    schema.addField("gravity", type="D")
    schema.addField("sed", type="D")
    schema.addField("dust", type="D")
    schema.getAliasMap().set('slot_Centroid', name + '_Centroid')

    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    star_properties = stellar_distribution(seed=seed, n_star=n_star, **kwargs)
    temperature = star_properties[0]
    flux = star_properties[1]
    metallicity = star_properties[2]
    surface_gravity = star_properties[3]
    rand_gen = np.random
    if seed is not None:
        rand_gen.seed(seed + 1)  # ensure that we use a different seed than stellar_distribution.
    x = rand_gen.uniform(x0 + edge_distance, x0 + x_size - edge_distance, n_star)
    y = rand_gen.uniform(y0 + edge_distance, y0 + y_size - edge_distance, n_star)

    catalog = afwTable.SourceCatalog(schema)
    fluxKey = schema.find(fluxName).key
    flagKey = schema.find(flagName).key
    fluxSigmaKey = schema.find(fluxSigmaName).key
    temperatureKey = schema.find("temperature").key
    metalKey = schema.find("metallicity").key
    gravityKey = schema.find("gravity").key
    centroidKey = afwTable.Point2DKey(schema["slot_Centroid"])
    for _i in range(n_star):
        source_test_centroid = afwGeom.Point2D(x[_i], y[_i])
        source = catalog.addNew()
        source.set(fluxKey, flux[_i])
        source.set(centroidKey, source_test_centroid)
        source.set(fluxSigmaKey, 0.)
        source.set(temperatureKey, temperature[_i])
        source.set(metalKey, metallicity[_i])
        source.set(gravityKey, surface_gravity[_i])
        source.set(flagKey, False)
    return(catalog.copy(True))  # Return a copy to make sure it is contiguous in memory.


def star_gen(sed_list=None, seed=None, temperature=5600, metallicity=0.0, surface_gravity=1.0,
             flux=1.0, bandpass=None):
    """!Generate a randomized spectrum at a given temperature over a range of wavelengths."""
    """
        Either use a supplied list of SEDs to be drawn from, or use a blackbody radiation model.
        The output is normalized to sum to the given flux.
        [future] If a seed is supplied, noise can be added to the final spectrum before normalization.
    """
    flux_to_jansky = 1.0e26
    f0 = constants.speed_of_light / (bandpass.wavelen_min * 1.0e-9)
    f1 = constants.speed_of_light / (bandpass.wavelen_max * 1.0e-9)
    f_cen = constants.speed_of_light / (bandpass.calc_eff_wavelen() * 1.0e-9)
    bandwidth_hz = f_cen * 2.0 * (f0 - f1) / (f0 + f1)

    def integral(generator):
        """Simple wrapper to make the math more apparent."""
        return(np.sum(var for var in generator))
    if sed_list is None:
        print("No sed_list supplied, using blackbody radiation spectra.")
        t_ref = [np.Inf, 0.0]
    else:
        temperature_list = [star.temp for star in sed_list]
        t_ref = [np.min(temperature_list), np.max(temperature_list)]

    bp_wavelen, bandpass_vals = bandpass.getBandpass()
    bandpass_gen = (bp for bp in bandpass_vals)
    bandpass_gen2 = (bp2 for bp2 in bandpass_vals)

    # If the desired temperature is outside of the range of models in sed_list, then use a blackbody.
    if temperature >= t_ref[0] and temperature <= t_ref[1]:
        temp_weight = [np.abs(t / temperature - 1.0) for t in temperature_list]
        temp_thresh = np.min(temp_weight)
        t_inds = np.where(temp_weight <= temp_thresh)
        t_inds = t_inds[0]  # unpack tuple from np.where()
        n_inds = len(t_inds)
        if n_inds > 1:
            grav_list = [sed_list[_i].logg for _i in t_inds]
            metal_list = [sed_list[_i].logZ for _i in t_inds]
            offset = 10.0  # Add an offset to the values to prevent dividing by zero
            grav_weight = (((grav + offset) / (surface_gravity + offset) - 1.0)**2 for grav in grav_list)
            metal_weight = (((metal + offset) / (metallicity + offset) - 1.0)**2 for metal in metal_list)
            composite_weight = [grav + metal for (grav, metal) in zip(grav_weight, metal_weight)]
            sed_i = t_inds[np.argmin(composite_weight)]
        else:
            sed_i = t_inds[0]

        def sed_integrate(sed=sed_list[sed_i], wave_start=None, wave_end=None):
            wavelengths = sed.wavelen
            flambdas = sed.flambda
            return(integral((flambdas[_i] for _i in range(len(flambdas))
                   if wavelengths[_i] >= wave_start and wavelengths[_i] < wave_end)))

        # integral over the full sed, to convert from W/m**2 to W/m**2/Hz
        sed_full_integral = sed_integrate(wave_end=np.Inf)
        flux_band_fraction = sed_integrate(wave_start=bandpass.wavelen_min, wave_end=bandpass.wavelen_max)
        flux_band_fraction /= sed_full_integral

        # integral over the full bandpass, to convert back to astrophysical quantities
        sed_band_integral = 0.0
        for wave_start, wave_end in wavelength_iterator(bandpass):
            sed_band_integral += next(bandpass_gen2) * sed_integrate(wave_start=wave_start, wave_end=wave_end)
        flux_band_norm = flux_to_jansky * flux * flux_band_fraction / bandwidth_hz

        for wave_start, wave_end in wavelength_iterator(bandpass):
            yield(flux_band_norm * next(bandpass_gen)
                  * sed_integrate(wave_start=wave_start, wave_end=wave_end) / sed_band_integral)

    else:
        h = constants.Planck
        kb = constants.Boltzmann
        c = constants.speed_of_light

        prefactor = 2.0 * (kb * temperature)**4. / (h**3 * c**2)

        def radiance_expansion(x, nterms):
            for n in range(1, nterms + 1):
                poly_term = x**3 / n + 3 * x**2 / n**2 + 6 * x / n**3 + 6 / n**4
                exp_term = np.exp(-n * x)
                yield(poly_term * exp_term)

        def radiance_calc(wavelength_start, wavelength_end, temperature=temperature, nterms=3):
            nu1 = c / (wavelength_start / 1E9)
            nu2 = c / (wavelength_end / 1E9)
            x1 = h * nu1 / (kb * temperature)
            x2 = h * nu2 / (kb * temperature)
            radiance1 = radiance_expansion(x1, nterms)
            radiance2 = radiance_expansion(x2, nterms)
            radiance_integral1 = prefactor * integral(radiance1)
            radiance_integral2 = prefactor * integral(radiance2)
            return(radiance_integral1 - radiance_integral2)

        # integral over the full sed, to convert from W/m**2 to W/m**2/Hz
        radiance_full_integral = radiance_calc(bandpass.wavelen_min / 100.0, bandpass.wavelen_max * 100.0)
        flux_band_fraction = radiance_calc(bandpass.wavelen_min, bandpass.wavelen_max)
        flux_band_fraction /= radiance_full_integral

        radiance_band_integral = 0.0
        for wave_start, wave_end in wavelength_iterator(bandpass):
            radiance_band_integral += next(bandpass_gen2) * radiance_calc(wave_start, wave_end)
        flux_band_norm = flux_to_jansky * flux * flux_band_fraction / bandwidth_hz

        for wave_start, wave_end in wavelength_iterator(bandpass):
            yield(flux_band_norm * next(bandpass_gen)
                  * radiance_calc(wave_start, wave_end) / radiance_band_integral)


def load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                  use_filter=True, use_detector=True, **kwargs):
    """!Load in Bandpass object from sims_photUtils."""
    class BandpassMod(Bandpass):
        """Customize a few methods of the Bandpass class from sims_photUtils."""

        def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
            """Calculate effective wavelengths for filters."""
            # This is useful for summary numbers for filters.
            # Calculate effective wavelength of filters.
            if self.phi is None:
                self.sbTophi()
            if wavelength_min is None:
                wavelength_min = np.min(self.wavelen)
            if wavelength_max is None:
                wavelength_max = np.max(self.wavelen)
            w_inds = (self.wavelen >= wavelength_min) & (self.wavelen <= wavelength_max)
            effwavelenphi = (self.wavelen[w_inds] * self.phi[w_inds]).sum() / self.phi[w_inds].sum()
            return effwavelenphi

    """Define the wavelength range and resolution for a given ugrizy band."""
    band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                 'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
    band_range = band_dict[band_name]
    bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1], wavelen_step=wavelength_step)
    throughput_dir = getPackageDir('throughputs')
    lens_list = ['baseline/lens1.dat', 'baseline/lens2.dat', 'baseline/lens3.dat']
    mirror_list = ['baseline/m1.dat', 'baseline/m2.dat', 'baseline/m3.dat']
    atmos_list = ['atmos/atmos_11.dat']
    detector_list = ['baseline/detector.dat']
    filter_list = ['baseline/filter_' + band_name + '.dat']
    component_list = []
    if use_mirror:
        component_list += mirror_list
    if use_lens:
        component_list += lens_list
    if use_atmos:
        component_list += atmos_list
    if use_detector:
        component_list += detector_list
    if use_filter:
        component_list += filter_list
    bandpass.readThroughputList(rootDir=throughput_dir, componentList=component_list)
    # Calculate bandpass phi value if required.
    if bandpass.phi is None:
        bandpass.sbTophi()
    return(bandpass)


def wavelength_iterator(bandpass, use_midpoint=False):
    """Define iterator to ensure that loops over wavelength are consistent."""
    wave_start = bandpass.wavelen_min
    while wave_start < bandpass.wavelen_max:
        wave_end = wave_start + bandpass.wavelen_step
        if wave_end > bandpass.wavelen_max:
            wave_end = bandpass.wavelen_max
        if use_midpoint:
            yield(bandpass.calc_eff_wavelen(wavelength_min=wave_start, wavelength_max=wave_end))
        else:
            yield((wave_start, wave_end))
        wave_start = wave_end


def stellar_distribution(seed=None, n_star=None, hottest_star='A', coolest_star='M', verbose=True, **kwargs):
    """!Function that attempts to return a realistic distribution of stellar properties.
    Returns temperature, flux, metallicity, surface gravity
    temperature in units Kelvin
    flux in units W/m**2
    metallicity is logarithmic metallicity relative to solar
    surface gravity relative to solar
    """
    star_prob = [76.45, 12.1, 7.6, 3, 0.6, 0.13, 3E-5]
    # Relative to Solar luminosity. Hotter stars are brighter on average.
    luminosity_scale = [(0.01, 0.08), (0.08, 0.6), (0.6, 1.5), (1.5, 5.0), (5.0, 100.0), (100.0, 30000.0),
                        (30000.0, 50000.0)]
    temperature_range = [(2400, 3700), (3700, 5200), (5200, 6000), (6000, 7500), (7500, 10000),
                         (10000, 30000), (30000, 50000)]  # in degrees Kelvin
    metallicity_range = [(-3.0, 0.5)] * len(star_prob)  # Assign a random log metallicity to each star.
    surface_gravity_range = [(0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.5, 2.0),
                             (1.0, 2.5), (2.0, 4.0), (3.0, 5.0)]
    lum_solar = 3.846e26  # Solar luminosity, in Watts
    ly = 9.4607e15  # one light year, in meters
    pi = np.pi
    luminosity_to_flux = lum_solar / (4.0 * pi * ly**2.0)
    star_type = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
    star_names = sorted(star_type.keys(), key=lambda star: star_type[star])
    s_hot = star_type[hottest_star] + 1
    s_cool = star_type[coolest_star]
    n_star_type = s_hot - s_cool
    star_prob = star_prob[s_cool:s_hot]
    star_prob.insert(0, 0)
    luminosity_scale = luminosity_scale[s_cool:s_hot]
    temperature_range = temperature_range[s_cool:s_hot]
    metallicity_range = metallicity_range[s_cool:s_hot]
    surface_gravity_range = surface_gravity_range[s_cool:s_hot]
    star_prob = np.cumsum(star_prob)
    max_prob = np.max(star_prob)
    rand_gen = np.random
    if seed is not None:
        rand_gen.seed(seed)
    star_sort = rand_gen.uniform(0, max_prob, n_star)
    temperature = []
    flux = []
    metallicity = []
    surface_gravity = []
    n_star = []
    flux_star = []
    for _i in range(n_star_type):
        inds = np.where((star_sort < star_prob[_i + 1]) * (star_sort > star_prob[_i]))
        inds = inds[0]  # np.where returns a tuple of two arrays
        n_star.append(len(inds))
        flux_stars_total = 0.0
        for ind in inds:
            temp_use = rand_gen.uniform(temperature_range[_i][0], temperature_range[_i][1])
            lum_use = rand_gen.uniform(luminosity_scale[_i][0], luminosity_scale[_i][1])
            # Assume that all stars are randomly distributed between 1 and 100 light years away
            distance_attenuation = rand_gen.uniform(1.0, 100.0) ** 2.0
            flux_use = lum_use * luminosity_to_flux / distance_attenuation
            metal_use = rand_gen.uniform(metallicity_range[_i][0], metallicity_range[_i][1])
            grav_use = rand_gen.uniform(surface_gravity_range[_i][0], surface_gravity_range[_i][1])
            temperature.append(temp_use)
            flux.append(flux_use)
            metallicity.append(metal_use)
            surface_gravity.append(grav_use)
            flux_stars_total += flux_use
        flux_star.append(flux_stars_total)
    flux_total = np.sum(flux_star)
    flux_star = [100. * _f / flux_total for _f in flux_star]
    info_string = "Number and flux contribution of stars of each type:\n"
    for _i in range(n_star_type):
        info_string += str(" [%s %i| %0.2f%%]" % (star_names[_i], n_star[_i], flux_star[_i]))
    if verbose:
        print(info_string)

    return((temperature, flux, metallicity, surface_gravity))
