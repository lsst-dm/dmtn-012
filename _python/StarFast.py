"""
StarFast is a simple but fast simulation tool to generate images for testing algorithms.
It is optimized for creating many realizations of the same field of sky under different observing conditions.

Four steps to set up the simulation:
1) Create a StarSim object
    example_sim = StarSim(options=options)
2) Load a PSF with example_sim.load_psf(psf, options=options)
    Example psf:
        import galsim
        gsp = galsim.GSParams(folding_threshold=1.0/x_size, maximum_fft_size=12288)
        psf = galsim.Kolmogorov(fwhm=1.0, flux=1, gsparams=gsp)
3) Create a simulated catalog of stars
    example_sim.load_catalog(options=options)
4) Build the raw sky model from the catalog
    example_sim.simulate()

The sky model can be used for many simulations of the same field under different conditions
For each simulated observation convolve the raw sky model with the PSF, and include instrumental,
atmospheric, etc... effects. If desired, a different psf may be supplied for each simulation.
    observed_image = example_sim.convolve(psf=psf, options=options)
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
from scipy import constants
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.sims.photUtils import Bandpass, matchStar
from lsst.utils import getPackageDir
from calc_refractive_index import diff_refraction
from fast_dft import fast_dft
import time
import unittest
import lsst.utils.tests as utilsTests


class StarSim:
    """Class that defines a random simulated region of sky, and allows fast transformations."""

    def __init__(self, psf=None, pixel_scale=0.25, pad_image=1.5, catalog=None, sed_list=None,
                 x_size=512, y_size=512, band_name='g', photons_per_adu=1e4, **kwargs):
        """Set up the fixed parameters of the simulation."""
        bandpass = _load_bandpass(band_name=band_name, **kwargs)
        self.n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.bandpass = bandpass
        if sed_list is None:
            # Load in model SEDs
            matchStarObj = matchStar()
            sed_list = matchStarObj.loadKuruczSEDs()
        self.sed_list = sed_list
        self.catalog = catalog
        self.coord = _CoordsXY(pixel_scale=pixel_scale, pad_image=pad_image, x_size=x_size, y_size=y_size)
        self.edge_dist = None
        self.kernel_radius = None
        if psf is not None:
            self.load_psf(psf, **kwargs)
        self.source_model = None
        self.bright_model = None
        self.n_star = None
        self.photons_per_adu = photons_per_adu  # used to approximate the effect of photon shot noise.

    def load_psf(self, psf, edge_dist=None, kernel_radius=None, **kwargs):
        """Load a PSF class from galsim. The class needs to have two methods, getFWHM() and drawImage()."""
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2. * np.log(2)))
        self.psf = psf
        CoordsXY = self.coord
        kernel_min_radius = np.ceil(5 * psf.getFWHM() * fwhm_to_sigma / CoordsXY.scale())
        if self.kernel_radius < kernel_min_radius:
            self.kernel_radius = kernel_min_radius
        if self.edge_dist is None:
            if CoordsXY.pad > 1:
                self.edge_dist = 0
            else:
                self.edge_dist = 5 * psf.getFWHM() * fwhm_to_sigma / CoordsXY.scale()

    def load_catalog(self, name=None, sed_list=None, n_star=None, **kwargs):
        """Load or generate a catalog of stars to be used for the simulations."""
        bright_sigma_threshold = 3.0
        bright_flux_threshold = 0.1
        CoordsXY = self.coord
        self.catalog = _cat_sim(x_size=CoordsXY.xsize(base=True), y_size=CoordsXY.ysize(base=True),
                                name=name, n_star=n_star, pixel_scale=CoordsXY.pixel_scale,
                                edge_distance=self.edge_dist, **kwargs)
        schema = self.catalog.getSchema()
        n_star = len(self.catalog)
        self.n_star = n_star
        if sed_list is None:
            sed_list = self.sed_list

        if name is None:
            # If no name is supplied, find the first entry in the schema in the format *_flux
            schema_entry = schema.extract("*_flux", ordered='true')
            fluxName = schema_entry.iterkeys().next()
        else:
            fluxName = name + '_flux'

        fluxKey = schema.find(fluxName).key
        temperatureKey = schema.find("temperature").key
        metalKey = schema.find("metallicity").key
        gravityKey = schema.find("gravity").key
        # if catalog.isContiguous()
        flux = self.catalog[fluxKey] / self.psf.getFlux()
        temperatures = self.catalog[temperatureKey]
        metallicities = self.catalog[metalKey]
        gravities = self.catalog[gravityKey]
        xv = self.catalog.getX()
        yv = self.catalog.getY()
        flux_arr = np.zeros((self.n_star, self.n_step))

        for _i in range(n_star):
            f_star = flux[_i]
            t_star = temperatures[_i]
            z_star = metallicities[_i]
            g_star = gravities[_i]
            star_spectrum = _star_gen(sed_list=sed_list, temperature=t_star, flux=f_star,
                                      bandpass=self.bandpass, metallicity=z_star, surface_gravity=g_star)
            flux_arr[_i, :] = np.array([flux_val for flux_val in star_spectrum])
        flux_tot = np.sum(flux_arr, axis=1)
        if n_star > 3:
            cat_sigma = np.std(flux_tot[flux_tot - np.median(flux_tot)
                                        < bright_sigma_threshold * np.std(flux_tot)])
            bright_inds = (np.where(flux_tot - np.median(flux_tot) > bright_sigma_threshold * cat_sigma))[0]
            if len(bright_inds) > 0:
                flux_faint = np.sum(flux_arr) - np.sum(flux_tot[bright_inds])
                bright_inds = [i_b for i_b in bright_inds
                               if flux_tot[i_b] > bright_flux_threshold * flux_faint]
            bright_flag = np.zeros(n_star)
            bright_flag[bright_inds] = 1
        else:
            bright_flag = np.ones(n_star)

        self.n_star = n_star
        self.star_flux = flux_arr
        CoordsXY.set_flag(bright_flag == 1)
        CoordsXY.set_x(xv)
        CoordsXY.set_y(yv)

    def simulate(self, verbose=True, **kwargs):
        """Call fast_dft.py to construct the input sky model for each frequency slice prior to convolution."""
        CoordsXY = self.coord
        n_bright = CoordsXY.n_flag()
        n_faint = self.n_star - n_bright
        bright_flag = True
        if n_faint > 0:
            CoordsXY.set_oversample(1)
            flux = self.star_flux[CoordsXY.flag != bright_flag]
            timing_model = -time.time()
            self.source_model = fast_dft(flux, CoordsXY.x_loc(bright=False), CoordsXY.y_loc(bright=False),
                                         x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                         kernel_radius=self.kernel_radius, no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_faint, bright=False, timing=timing_model))
        if n_bright > 0:
            CoordsXY.set_oversample(2)
            flux = self.star_flux[CoordsXY.flag == bright_flag]
            timing_model = -time.time()
            self.bright_model = fast_dft(flux, CoordsXY.x_loc(bright=True), CoordsXY.y_loc(bright=True),
                                         x_size=CoordsXY.xsize(), y_size=CoordsXY.ysize(),
                                         kernel_radius=CoordsXY.xsize(), no_fft=False, **kwargs)
            timing_model += time.time()
            if verbose:
                print(_timing_report(n_star=n_bright, bright=True, timing=timing_model))

    def convolve(self, seed=None, sky_noise=0, instrument_noise=0, photon_noise=0, verbose=True, **kwargs):
        """Convolve a simulated sky with a given PSF."""
        CoordsXY = self.coord
        sky_noise_gen = _sky_noise_gen(CoordsXY, seed=seed, amplitude=sky_noise,
                                       n_step=self.n_step, verbose=verbose)
        if self.source_model is not None:
            source_image = self._convolve_subroutine(sky_noise_gen, verbose=verbose, bright=False, **kwargs)
        else:
            source_image = 0.0
        if self.bright_model is not None:
            bright_image = self._convolve_subroutine(sky_noise_gen, verbose=verbose, bright=True, **kwargs)
        else:
            bright_image = 0.0
        return_image = source_image + bright_image

        if photon_noise > 0:
            rand_gen = np.random
            if seed is not None:
                rand_gen.seed(seed - 1.2)
            sky_photons = np.sqrt(np.abs(return_image * self.photons_per_adu)) / self.photons_per_adu
            return_image += sky_photons * rand_gen.normal(scale=photon_noise, size=return_image.shape)

        if instrument_noise > 0:
            rand_gen = np.random
            if seed is not None:
                rand_gen.seed(seed - 1.1)
            return_image += rand_gen.normal(scale=instrument_noise, size=return_image.shape)
        return(return_image)

    def _convolve_subroutine(self, sky_noise_gen, psf=None, verbose=True, bright=False, **kwargs):
        CoordsXY = self.coord
        if bright:
            CoordsXY.set_oversample(2)
        else:
            CoordsXY.set_oversample(1)
        dcr_gen = _dcr_generator(self.bandpass, pixel_scale=CoordsXY.scale(), **kwargs)
        convol = np.zeros((CoordsXY.ysize(), CoordsXY.xsize() // 2 + 1), dtype='complex64')
        if psf is None:
            psf = self.psf
        if self.psf is None:
            self.load_psf(psf)
        timing_fft = -time.time()

        for _i, offset in enumerate(dcr_gen):
            if bright:
                source_model_use = self.bright_model[_i]
            else:
                source_model_use = self.source_model[_i]

            psf_image = psf.drawImage(scale=CoordsXY.scale(), method='fft', offset=offset,
                                      nx=CoordsXY.xsize(), ny=CoordsXY.ysize(), use_true_center=False)
            try:
                #  Note: if adding sky noise, it should only added once (check if the generator is exhausted)
                source_model_use += next(sky_noise_gen)
            except StopIteration:
                pass
            convol_single = source_model_use * rfft2(psf_image.array)
            convol += convol_single
        return_image = np.real(fftshift(irfft2(convol))) * CoordsXY.oversample**2.0
        timing_fft += time.time()
        if verbose:
            print("FFT timing for %i DCR planes: [%0.3fs | %0.3fs per plane]"
                  % (self.n_step, timing_fft, timing_fft / self.n_step))
        return_image = return_image[CoordsXY.ymin():CoordsXY.ymax():CoordsXY.oversample,
                                    CoordsXY.xmin():CoordsXY.xmax():CoordsXY.oversample]
        if bright:
            CoordsXY.set_oversample(1)
        return(return_image)


def _sky_noise_gen(CoordsXY, seed=None, amplitude=None, n_step=1, verbose=False):
    """Generate random sky noise in Fourier space."""
    if amplitude > 0:
        if verbose:
            print("Adding sky noise with amplitude %f" % amplitude)
        rand_gen = np.random
        if seed is not None:
            rand_gen.seed(seed - 1)
        #  Note: it's important to use CoordsXY.xsize() here, since CoordsXY is updated for bright stars
        y_size_use = CoordsXY.ysize()
        x_size_use = CoordsXY.xsize() // 2 + 1
        amplitude_use = amplitude / (np.sqrt(n_step / (x_size_use * y_size_use)))
        for _i in range(n_step):
            rand_fft = (rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use))
                        + 1j * rand_gen.normal(scale=amplitude_use, size=(y_size_use, x_size_use)))
            yield(rand_fft)


class _CoordsXY:
    def __init__(self, pad_image=1.5, pixel_scale=None, x_size=None, y_size=None):
        self._x_size = x_size
        self._y_size = y_size
        self.pixel_scale = pixel_scale
        self.oversample = 1
        self.pad = pad_image
        self.flag = None

    def set_x(self, x_loc):
        self._x = x_loc

    def set_y(self, y_loc):
        self._y = y_loc

    def set_flag(self, flag):
        self.flag = flag.astype(bool)

    def n_flag(self):
        if self.flag is None:
            n = 0
        else:
            n = np.sum(self.flag)
        return(n)

    def set_oversample(self, oversample):
        self.oversample = int(oversample)

    def xsize(self, base=False):
        if base:
            return(int(self._x_size))
        else:
            return(int(self._x_size * self.pad) * self.oversample)

    def xmin(self):
        return(int(self.oversample * (self._x_size * (self.pad - 1) // 2)))

    def xmax(self):
        return(int(self.xmin() + self._x_size * self.oversample))

    def ysize(self, base=False):
        if base:
            return(int(self._y_size))
        else:
            return(int(self._y_size * self.pad) * self.oversample)

    def ymin(self):
        return(int(self.oversample * (self._y_size * (self.pad - 1) // 2)))

    def ymax(self):
        return(int(self.ymin() + self._y_size * self.oversample))

    def scale(self):
        return(self.pixel_scale / self.oversample)

    def x_loc(self, bright=False):
        x_loc = self._x * self.oversample + self.xmin()
        if self.flag is not None:
            x_loc = x_loc[self.flag == bright]
        return(x_loc)

    def y_loc(self, bright=False):
        y_loc = self._y * self.oversample + self.ymin()
        if self.flag is not None:
            y_loc = y_loc[self.flag == bright]
        return(y_loc)


def _timing_report(n_star=None, bright=False, timing=None):
    if bright:
        bright_star = "bright "
    else:
        bright_star = ""
    if n_star == 1:
        return("Time to model %i %sstar: [%0.3fs]" % (n_star, bright_star, timing))
    else:
        return("Time to model %i %sstars: [%0.3fs | %0.5fs per star]"
               % (n_star, bright_star, timing, timing / n_star))


def _dcr_generator(bandpass, pixel_scale=None, elevation=None, azimuth=None, **kwargs):
    """!Call the functions that compute Differential Chromatic Refraction (relative to mid-band)."""
    if elevation is None:
        elevation = 50.0
    if azimuth is None:
        azimuth = 0.0
    zenith_angle = 90.0 - elevation
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in _wavelength_iterator(bandpass, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = diff_refraction(wavelength=wavelength, wavelength_ref=wavelength_midpoint,
                                      zenith_angle=zenith_angle, **kwargs)
        refract_amp *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
        dx = refract_amp * np.sin(np.radians(azimuth))
        dy = refract_amp * np.cos(np.radians(azimuth))
        yield((dx, dy))


def _cat_sim(x_size=None, y_size=None, seed=None, n_star=None, n_galaxy=None,
             edge_distance=0, name=None, pixel_scale=None, **kwargs):
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

    x_size_gen = x_size - 2 * edge_distance
    y_size_gen = y_size - 2 * edge_distance
    star_properties = _stellar_distribution(seed=seed, n_star=n_star, pixel_scale=pixel_scale,
                                            x_size=x_size_gen, y_size=y_size_gen, **kwargs)
    temperature = star_properties[0]
    flux = star_properties[1]
    metallicity = star_properties[2]
    surface_gravity = star_properties[3]
    x = star_properties[4]
    y = star_properties[5]

    catalog = afwTable.SourceCatalog(schema)
    fluxKey = schema.find(fluxName).key
    flagKey = schema.find(flagName).key
    fluxSigmaKey = schema.find(fluxSigmaName).key
    temperatureKey = schema.find("temperature").key
    metalKey = schema.find("metallicity").key
    gravityKey = schema.find("gravity").key
    centroidKey = afwTable.Point2DKey(schema["slot_Centroid"])
    for _i in range(n_star):
        source_test_centroid = afwGeom.Point2D(x[_i] + edge_distance, y[_i] + edge_distance)
        source = catalog.addNew()
        source.set(fluxKey, flux[_i])
        source.set(centroidKey, source_test_centroid)
        source.set(fluxSigmaKey, 0.)
        source.set(temperatureKey, temperature[_i])
        source.set(metalKey, metallicity[_i])
        source.set(gravityKey, surface_gravity[_i])
        source.set(flagKey, False)
    return(catalog.copy(True))  # Return a copy to make sure it is contiguous in memory.


def _star_gen(sed_list=None, seed=None, temperature=5600, metallicity=0.0, surface_gravity=1.0,
              flux=1.0, bandpass=None, verbose=True):
    """Generate a randomized spectrum at a given temperature over a range of wavelengths."""
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
        if verbose:
            print("No sed_list supplied, using blackbody radiation spectra.")
        t_ref = [np.Inf, 0.0]
    else:
        temperatures = np.array([star.temp for star in sed_list])
        t_ref = [temperatures.min(), temperatures.max()]

    bp_wavelen, bandpass_vals = bandpass.getBandpass()
    bandpass_gen = (bp for bp in bandpass_vals)
    bandpass_gen2 = (bp2 for bp2 in bandpass_vals)

    # If the desired temperature is outside of the range of models in sed_list, then use a blackbody.
    if temperature >= t_ref[0] and temperature <= t_ref[1]:
        temp_weight = np.abs(temperatures / temperature - 1.0)
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
            waves = (wavelengths >= wave_start) & (wavelengths < wave_end)
            return(flambdas[waves].sum())

        # integral over the full sed, to convert from W/m**2 to W/m**2/Hz
        sed_full_integral = sed_integrate(wave_end=np.Inf)
        flux_band_fraction = sed_integrate(wave_start=bandpass.wavelen_min, wave_end=bandpass.wavelen_max)
        flux_band_fraction /= sed_full_integral

        # integral over the full bandpass, to convert back to astrophysical quantities
        sed_band_integral = 0.0
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            sed_band_integral += (next(bandpass_gen2)
                                  * sed_integrate(wave_start=wave_start, wave_end=wave_end))
        flux_band_norm = flux_to_jansky * flux * flux_band_fraction / bandwidth_hz

        for wave_start, wave_end in _wavelength_iterator(bandpass):
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
        for wave_start, wave_end in _wavelength_iterator(bandpass):
            radiance_band_integral += next(bandpass_gen2) * radiance_calc(wave_start, wave_end)
        flux_band_norm = flux_to_jansky * flux * flux_band_fraction / bandwidth_hz

        for wave_start, wave_end in _wavelength_iterator(bandpass):
            yield(flux_band_norm * next(bandpass_gen)
                  * radiance_calc(wave_start, wave_end) / radiance_band_integral)


def _load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
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
    bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1],
                           wavelen_step=wavelength_step)
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


def _wavelength_iterator(bandpass, use_midpoint=False):
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


def _stellar_distribution(seed=None, n_star=None, hottest_star='A', coolest_star='M',
                          x_size=None, y_size=None, pixel_scale=None, verbose=True, **kwargs):
    """!Function that attempts to return a realistic distribution of stellar properties.
    Returns temperature, flux, metallicity, surface gravity
    temperature in units Kelvin
    flux in units W/m**2
    metallicity is logarithmic metallicity relative to solar
    surface gravity relative to solar
    """
    # Percent abundance of stellar types M,K,G,F,A,B,O 
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
    pixel_scale_degrees = pixel_scale / 3600.0
    max_star_dist = 1000  # light years
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
    x_star = []
    y_star = []
    z_star = []
    x_scale = np.sin(np.radians(x_size * pixel_scale_degrees)) / 2
    y_scale = np.sin(np.radians(y_size * pixel_scale_degrees)) / 2
    for _i in range(n_star_type):
        inds = np.where((star_sort < star_prob[_i + 1]) * (star_sort > star_prob[_i]))
        inds = inds[0]  # np.where returns a tuple of two arrays
        n_star.append(len(inds))
        flux_stars_total = 0.0
        for ind in inds:
            temp_use = rand_gen.uniform(temperature_range[_i][0], temperature_range[_i][1])
            lum_use = rand_gen.uniform(luminosity_scale[_i][0], luminosity_scale[_i][1])
            bounds_test = True
            while bounds_test:
                x_dist = rand_gen.uniform(-max_star_dist * x_scale, max_star_dist * x_scale)
                z_dist = rand_gen.uniform(1.0, max_star_dist)
                if np.abs(x_dist) < x_scale * z_dist:
                    y_dist = rand_gen.uniform(-max_star_dist * y_scale, max_star_dist * y_scale)
                    if np.abs(y_dist) < y_scale * z_dist:
                        bounds_test = False
            x_star.append(x_size / 2 + np.degrees(np.arctan(x_dist / z_dist)) / pixel_scale_degrees)
            y_star.append(y_size / 2 + np.degrees(np.arctan(y_dist / z_dist)) / pixel_scale_degrees)
            z_star.append(z_dist)
            distance_attenuation = z_dist ** 2.0
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
        info_string += str(" [%s %i| %0.2f%%]" % (star_names[_i + s_cool], n_star[_i], flux_star[_i]))
    if verbose:
        print(info_string)
    return((temperature, flux, metallicity, surface_gravity, x_star, y_star))


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class _BasicBandpass:
    """Dummy bandpass object for testing."""

    def __init__(self, band_name='g', wavelength_step=1):
        """Define the wavelength range and resolution for a given ugrizy band."""
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[band_name]
        self.wavelen_min = band_range[0]
        self.wavelen_max = band_range[1]
        self.wavelen_step = wavelength_step

    def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
        """Mimic the calc_eff_wavelen method of the real bandpass class."""
        if wavelength_min is None:
            wavelength_min = self.wavelen_min
        if wavelength_max is None:
            wavelength_max = self.wavelen_max
        return((wavelength_min + wavelength_max) / 2.0)

    def getBandpass(self):
        """Mimic the getBandpass method of the real bandpass class."""
        wl_gen = _wavelength_iterator(self)
        wavelengths = [wl[0] for wl in wl_gen]
        wavelengths += [self.wavelen_max]
        bp_vals = [1] * len(wavelengths)
        return((wavelengths, bp_vals))


class _BasicSED:
    """Dummy SED for testing."""

    def __init__(self, temperature=5600.0, metallicity=0.0, surface_gravity=1.0):
        wavelen_min = 10.0
        wavelen_max = 2000.0
        wavelen_step = 1
        self.temp = temperature
        self.logg = surface_gravity
        self.logZ = metallicity
        self.wavelen = np.arange(wavelen_min, wavelen_max, wavelen_step)
        self.flambda = np.arange(wavelen_min, wavelen_max, wavelen_step) / wavelen_max


class CoordinatesTestCase(utilsTests.TestCase):
    """Test the simple coordinate transformation class."""

    def setUp(self):
        """Define parameters used by every test."""
        seed = 42
        rand_gen = np.random
        rand_gen.seed(seed)
        self.pixel_scale = 0.25
        self.pad_image = 1.5
        self.x_size = 128
        self.y_size = 128
        self.n_star = 30
        self.n_bright = 10
        self.x_loc = rand_gen.uniform(high=self.x_size, size=self.n_star)
        self.y_loc = rand_gen.uniform(high=self.y_size, size=self.n_star)
        self.flag_array = np.array([False] * self.n_star)
        self.flag_array[:2 * self.n_bright:2] = True
        self.coords = _CoordsXY(pixel_scale=self.pixel_scale, pad_image=self.pad_image,
                                x_size=self.x_size, y_size=self.y_size)

    def tearDown(self):
        """Clean up."""
        del self.coords
        del self.flag_array

    def test_coord_size_normal_scale(self):
        """Make sure everything gets set, and the math is correct."""
        self.assertAlmostEqual(self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_size_no_scale(self):
        """Make sure we can recover input dimensions."""
        self.assertAlmostEqual(self.x_size, self.coords.xsize(base=True))
        self.assertAlmostEqual(self.y_size, self.coords.ysize(base=True))

    def test_coord_size_over_scale(self):
        """Make sure everything gets set, and the math is correct."""
        self.coords.set_oversample(2)
        self.assertAlmostEqual(2 * self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2 * self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_size_over_scale_nonint(self):
        """Oversampling must only by integer factors."""
        self.coords.set_oversample(2.3)
        self.assertAlmostEqual(2 * self.pad_image * self.x_size, self.coords.xsize())
        self.assertAlmostEqual(2 * self.pad_image * self.y_size, self.coords.ysize())

    def test_coord_pixel_scale_base(self):
        """Make sure everything gets set, and the math is correct."""
        self.assertEqual(self.pixel_scale, self.coords.scale())

    def test_coord_pixel_scale_over(self):
        """Make sure everything gets set, and the math is correct."""
        self.coords.set_oversample(2)
        self.assertEqual(self.pixel_scale / 2, self.coords.scale())

    def test_bright_count(self):
        """Check that the number of locations flagged as bright is correct."""
        self.coords.set_flag(self.flag_array)
        self.assertEqual(self.n_bright, self.coords.n_flag())

    def test_faint_source_locations(self):
        """Check that locations of faint sources are computed correctly, and that flags are correct."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        CoordsXY.set_flag(self.flag_array)
        bright_condition = True
        faint_x = self.x_loc[self.flag_array != bright_condition]
        faint_y = self.y_loc[self.flag_array != bright_condition]
        abs_diff_x = np.sum(np.abs(faint_x + CoordsXY.xmin() - CoordsXY.x_loc()))
        abs_diff_y = np.sum(np.abs(faint_y + CoordsXY.ymin() - CoordsXY.y_loc()))
        self.assertAlmostEqual(abs_diff_x, 0)
        self.assertAlmostEqual(abs_diff_y, 0)

    def test_bright_source_locations(self):
        """Check that locations of bright sources are computed correctly, and that flags are correct."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        CoordsXY.set_flag(self.flag_array)
        CoordsXY.set_oversample(2)
        bright_condition = True
        bright_x = 2 * self.x_loc[self.flag_array == bright_condition]
        bright_y = 2 * self.y_loc[self.flag_array == bright_condition]
        abs_diff_x = np.sum(np.abs(bright_x + CoordsXY.xmin() - CoordsXY.x_loc(bright=True)))
        abs_diff_y = np.sum(np.abs(bright_y + CoordsXY.ymin() - CoordsXY.y_loc(bright=True)))
        self.assertAlmostEqual(abs_diff_x, 0)
        self.assertAlmostEqual(abs_diff_y, 0)

    def test_faint_sources_no_flags(self):
        """If there are no flags, all source locations should always be returned."""
        CoordsXY = self.coords
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        self.assertEqual(len(CoordsXY.x_loc()), self.n_star)
        self.assertEqual(len(CoordsXY.y_loc()), self.n_star)

    def test_bright_sources_no_flags(self):
        """If there are no flags, all source locations should always be returned."""
        CoordsXY = self.coords
        CoordsXY.set_oversample(2)
        CoordsXY.set_x(self.x_loc)
        CoordsXY.set_y(self.y_loc)
        self.assertEqual(len(CoordsXY.x_loc(bright=True)), self.n_star)
        self.assertEqual(len(CoordsXY.y_loc(bright=True)), self.n_star)


class DCRTestCase(utilsTests.TestCase):
    """Test the the calculations of Differential Chromatic Refraction."""

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        wavelength_step = 10.0
        self.pixel_scale = 0.25
        self.bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step)

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_dcr_generator(self):
        """Check that _dcr_generator returns a generator with n_step iterations, and (0,0) at zenith."""
        azimuth = 0.0
        elevation = 90.0
        zenith_dcr = (0.0, 0.0)
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            self.assertAlmostEqual(next(dcr_gen), zenith_dcr)
        with self.assertRaises(StopIteration):
            next(dcr_gen)

    def test_dcr_values(self):
        """Check DCR against pre-computed values."""
        azimuth = 0.0
        elevation = 50.0
        dcr_vals = [1.73959243097, 1.44317957935, 1.1427147535, 0.864107322861, 0.604249563363,
                    0.363170721045, 0.137678490152, -0.0730964797295, -0.270866384702, -0.455135994183,
                    -0.628721688199, -0.791313886049, -0.946883455499, -1.08145326102, -1.16120917137]
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            self.assertAlmostEqual(next(dcr_gen)[1], dcr_vals[_i])


class BandpassTestCase(utilsTests.TestCase):
    """Tests of the interface to Bandpass from lsst.sims.photUtils."""

    def setUp(self):
        """Define parameters used by every test."""
        self.band_name = 'g'
        self.wavelength_step = 10
        self.bandpass = _load_bandpass(band_name=self.band_name, wavelength_step=self.wavelength_step)

    def test_step_bandpass(self):
        """Check that the bandpass has necessary methods, and those return the correct number of values."""
        bp = self.bandpass
        bp_wavelen, bandpass_vals = bp.getBandpass()
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))


class StarGenTestCase(utilsTests.TestCase):
    """Test the flux calculation for a single star."""

    def setUp(self):
        """Define parameters used by every test."""
        self.bandpass = _BasicBandpass(band_name='g', wavelength_step=10)
        self.flux = 1E-9

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_blackbody_spectrum(self):
        """Check the blackbody spectrum against pre-computed values."""
        star_gen = _star_gen(temperature=5600, flux=self.flux, bandpass=self.bandpass, verbose=False)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([5.763797967, 5.933545118, 6.083468705, 6.213969661,
                                      6.325613049, 6.419094277, 6.495208932, 6.554826236,
                                      6.598866015, 6.628278971, 6.644030031, 6.647084472,
                                      6.638396542, 6.618900302, 4.616292143])
        abs_diff_spectrum = np.sum(np.abs(spectrum - pre_comp_spectrum))
        self.assertAlmostEqual(abs_diff_spectrum, 0.0)

    def test_sed_spectrum(self):
        """Check a spectrum defined by an SED against pre-computed values."""
        temperature = 5600
        sed_list = [_BasicSED(temperature)]
        star_gen = _star_gen(sed_list=sed_list, temperature=temperature, flux=self.flux,
                             bandpass=self.bandpass, verbose=True)
        spectrum = np.array([flux for flux in star_gen])
        pre_comp_spectrum = np.array([1.06433106, 1.09032205, 1.11631304, 1.14230403, 1.16829502,
                                      1.19428601, 1.22027700, 1.24626799, 1.27225898, 1.29824997,
                                      1.32424096, 1.35023195, 1.37622294, 1.40221393, 0.99701439])
        abs_diff_spectrum = np.sum(np.abs(spectrum - pre_comp_spectrum))
        self.assertAlmostEqual(abs_diff_spectrum, 0.0)


class StellarDistributionTestCase(utilsTests.TestCase):
    """Verify that the random catalog generation is unchanged."""

    def setUp(self):
        """Define parameters used by every test."""
        self.x_size = 10
        self.y_size = 10
        self.pixel_scale = 0.25
        self.seed = 42

    def test_star_type_properties(self):
        """Check that the properties of stars of a given type all fall in the right ranges."""
        star_properties = _stellar_distribution(seed=self.seed, n_star=3, pixel_scale=self.pixel_scale,
                                                x_size=self.x_size, y_size=self.y_size,
                                                hottest_star='G', coolest_star='G', verbose=False)
        temperature = star_properties[0]
        metallicity = star_properties[2]
        surface_gravity = star_properties[3]
        temp_range_g_star = [5200, 6000]
        grav_range_g_star = [0.0, 1.5]
        metal_range_g_star = [-3.0, 0.5]
        self.assertLessEqual(np.max(temperature), temp_range_g_star[1])
        self.assertGreaterEqual(np.min(temperature), temp_range_g_star[0])

        self.assertLessEqual(np.max(surface_gravity), grav_range_g_star[1])
        self.assertGreaterEqual(np.min(surface_gravity), grav_range_g_star[0])

        self.assertLessEqual(np.max(metallicity), metal_range_g_star[1])
        self.assertGreaterEqual(np.min(metallicity), metal_range_g_star[0])

    def test_star_xy_range(self):
        """Check that star pixel coordinates are all in range."""
        star_properties = _stellar_distribution(seed=self.seed, n_star=3, pixel_scale=self.pixel_scale,
                                                x_size=self.x_size, y_size=self.y_size, verbose=False)
        x = star_properties[4]
        y = star_properties[5]
        self.assertLess(np.max(x), self.x_size)
        self.assertGreaterEqual(np.min(x), 0.0)

        self.assertLess(np.max(y), self.y_size)
        self.assertGreaterEqual(np.min(y), 0.0)


class SkyNoiseTestCase(utilsTests.TestCase):
    """Verify that the random catalog generation is unchanged."""

    def setUp(self):
        """Define parameters used by every test."""
        self.coord = _CoordsXY(pad_image=1, x_size=64, y_size=64)
        self.n_step = 3
        self.amplitude = 1.0
        self.seed = 3

    def tearDown(self):
        """Clean up."""
        del self.coord

    def test_noise_sigma(self):
        """The sky noise should be normalized such that the standard deviation of the image == amplitude."""
        CoordsXY = self.coord
        noise_gen = _sky_noise_gen(CoordsXY, seed=self.seed, amplitude=self.amplitude,
                                   n_step=self.n_step, verbose=False)
        noise_fft = next(noise_gen)
        for fft_single in noise_gen:
            noise_fft += fft_single
        noise_image = np.real(fftshift(irfft2(noise_fft)))
        dimension = np.sqrt(CoordsXY.xsize() * CoordsXY.ysize())
        self.assertLess(np.abs(np.std(noise_image) - self.amplitude), 1.0 / dimension)


def suite():
    """Return a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(CoordinatesTestCase)
    suites += unittest.makeSuite(DCRTestCase)
    suites += unittest.makeSuite(BandpassTestCase)
    suites += unittest.makeSuite(StarGenTestCase)
    suites += unittest.makeSuite(StellarDistributionTestCase)
    suites += unittest.makeSuite(SkyNoiseTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests."""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
