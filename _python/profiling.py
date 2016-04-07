#!/usr/bin/env python
"""Profiling StarFast"""

import os
# import cProfile
# import pstats
import cPickle

import galsim
from lsst.sims.photUtils import matchStar
from StarFast import star_sim

pickle_file = 'sed_list.pickle'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as dumpfile:
        sed_list = cPickle.load(dumpfile)
else:
    matchStarObj = matchStar()
    sed_list = matchStarObj.loadKuruczSEDs()
    with open(pickle_file, 'wb') as dumpfile:
        cPickle.dump(sed_list, dumpfile, cPickle.HIGHEST_PROTOCOL)

seed = 5
dimension = 1024  # number of pixels on a side of the simulation
pad_image = 1.5
n_star = 1000
pixel_scale = 0.25  # arcsec / pixel
band_name = 'u'  # vaild options are 'u', 'g', 'r', 'i', 'z', or 'y'
# vaild options are None (defaults to 'A') or any stellar type OBAFGKM hotter than coolest star
hottest_star = 'O'
# vaild options are None (defaults to 'M') or any stellar type OBAFGKM cooler than hottest star
coolest_star = 'K'
wavelength_step = 3  # wavelength bin size, in nm
photons_per_adu = 1e4
gsp = galsim.GSParams(folding_threshold=1.0/(dimension), maximum_fft_size=12288)
psf = galsim.Kolmogorov(fwhm=1.0, flux=1, gsparams=gsp)

ref_elevation = 85.0  # in degrees
ref_azimuth = 0.0  # in degrees

obs_elevation = 90.0 - 33.6  # in degrees
obs_azimuth = 20.0  # in degrees

sky_noise = 0.0
instrument_noise = 0.0
photon_noise = False

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This section for line_profiler: https://github.com/rkern/line_profiler
# add @profile decorators to the functions you want it to profile.
# Run with: kernprof -l profiling.py
# see results with "python -m line_profiler profiling.py.lprof"
star_sim(seed=seed, psf=psf, n_star=n_star, x_size=dimension, y_size=dimension,
         sky_noise=sky_noise, photon_noise=photon_noise,
         instrument_noise=instrument_noise, pixel_scale=pixel_scale, dcr_flag=False,
         elevation=ref_elevation, azimuth=ref_azimuth, band_name=band_name,
         wavelength_step=wavelength_step,
         hottest_star=hottest_star,
         coolest_star=coolest_star,
         sed_list=sed_list, pad_image=pad_image)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This section for cProfile.
# uncomment this block (and comment out the above) to produce a the listed profiling stats file.
# command = """ref_image = star_sim(seed=seed, psf=psf, n_star=n_star, x_size=dimension, y_size=dimension,
#                      sky_noise=sky_noise, photon_noise=photon_noise,
#                      instrument_noise=instrument_noise, pixel_scale=pixel_scale, dcr_flag=False,
#                      elevation=ref_elevation, azimuth=ref_azimuth, band_name=band_name,
#                      wavelength_step=wavelength_step,
#                      hottest_star=hottest_star,
#                      coolest_star=coolest_star,
#                      sed_list=sed_list, pad_image=pad_image)"""
# filename = 'ref_image.prof'
# cProfile.run(command, filename=filename)
# stats = pstats.Stats(filename)
# stats.strip_dirs()
# stats.sort_stats('cumtime').print_stats(20)
