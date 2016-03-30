..
  Content of technical report.

  See http://docs.lsst.codes/en/latest/development/docs/rst_styleguide.html
  for a guide to reStructuredText writing.

  Do not put the title, authors or other metadata in this document;
  those are automatically added.

  Use the following syntax for sections:

  Sections
  ========

  and

  Subsections
  -----------

  and

  Subsubsections
  ^^^^^^^^^^^^^^

  To add images, add the image file (png, svg or jpeg preferred) to the
  _static/ directory. The reST syntax for adding the image is

  .. figure:: /_static/filename.ext
     :name: fig-label
     :target: http://target.link/url

     Caption text.

   Run: ``make html`` and ``open _build/html/index.html`` to preview your work.
   See the README at https://github.com/lsst-sqre/lsst-report-bootstrap or
   this repo's README for more info.

   Feel free to delete this instructional comment.

:tocdepth: 2

Warning

This technical note is currently a draft!

#########
StarFast - A Fast Simulation Tool for Testing Algorithms
#########
(StarSim, SimFast, Cynthia, Simantha)

Overview
========
StarFast is a simple but fast simulation tool to generate images for testing algorithms. 
It has been developed primarily with an aim to quickly generate many images affected by realistic Differential Chromatic Refraction (DCR) and astrometric errors, but it is hopefully general enough to be used for other cases when we need to test algorithms against simulated data. 

It will generate a simulated catalog of stars (see :ref:`section-headings-sim-cat` below) and compute the flux in Janskys for each source given the instrument bandpass. 
For wavelength-dependent effects such as DCR, each band is divided into multiple sub-band planes. 
Sources are convolved with the PSF in Fourier space (but see :ref:`section-headings-implementation` below for details), and any sub-band planes are stacked to produce the final image. Noise may optionally be injected at any step, to simulate photon shot noise of the sources, flat spectrum sky noise, or receiver noise.

Several example images are shown below, which were produced using the included iPython notebook (in _python/sim_fast example).
Each image contains the same catalog of 10,000 stars, consisting of 7642 M, 1213 K, 751 G, 329 F, 60 A, and 5 B type stars (which can be obtained with the example notebook by setting seed = 5). 
For this simulation I have generated two LSST u-band images: a 'reference' image at an airmass of 1.00 near zenith that used a single plane, and a 'science' image at an airmass of 1.06 that used 23 planes (a wavelength resolution of 3nm).
Each star in the simulation uses a simulated Kurucz SED from Galsim, and a Kolmogorov PSF also from Galsim.
The reference image took just over 3 minutes to generate on a single core of a 2015 Macbook, while the the science image took roughly a minute longer despite having to simulate all 23 planes.
In this simulation, the brightest star is over seven orders of magnitude brighter than the faintest, so the reference image is displayed on a logarithmic color scale in Figure 1 and on a clipped linear color scale in Figure 2 below. The science image is not visually different from the reference image on either color scale so only the difference image is displayed in Figure 3.
Note that the very hot and bright B type stars have a DCR dipole in the opposite direction of the cooler stars, which is precisely what a DCR algorithm must be designed to correct.



.. figure:: /_static/ref_img10000_log.png
   :name: fig-ref-img-log
   :target: ../../_static/ref_img10000_log.png
   :alt: Simulated 1024x1024 image with 10,000 stars

   Simulated 1024x1024 image with 10,000 stars (logarithmic color scale).

.. figure:: /_static/ref_img10000_linear.png
   :name: fig-ref-img-linear
   :target: ../../_static/ref_img10000_linear.png
   :alt: Simulated 1024x1024 image with 10,000 stars

   As Figure 1, with a clipped linear color scale.

.. figure:: /_static/dcr_img10000_linear.png
   :name: fig-dcr-img-linear
   :target: ../../_static/dcr_img10000_linear.png
   :alt: Difference of two simulated images, with dipoles caused by DCR

   Difference of two simulated images of the same 10,000 stars, with the reference image at airmass 1.0 and the other at airmass 1.06.

.. _section-headings-sim-cat:

Simulated catalogs
==================
StarFast begins by creating a simulated catalog of stars with properties representative of the local population. 
Properties are drawn from random distributions for the stellar types O-M in proportion to their local abundance, and each star is set at a random distance from 1-100 light years. 
Simulated properties include temperature (in degrees K), luminosity (relative to solar), surface gravity (relative to solar), and metallicity (log, relative to solar). 
These properties are matched to Kurucz model SEDs from GalSim, which are scaled by the luminosity of each star and attenuated by distance and the instrument bandpass throughput to yield fluxes in Janskys. 
Alternately, a simple blackbody radiation spectrum (with the bandpass) can be used. 

Coordinates are also chosen from a random distribution, with no attempt at simulating realistic clustering. 
All of the random distributions, including stellar properties and coordinates, can be initialized from a user-supplied seed value, which allows for repeated simulations of the same patch of sky under different conditions. 
Simulated catalogs may also be returned and saved, so they may be modified by external tools if desired, and those saved catalogs may be supplied in place of generating a new random catalog from a seed.


.. _section-headings-implementation:

Implementation
==============


.. _section-headings-uses:

Use Cases
=========
Differential Chromatic Refraction (DCR)
---------------------------------------

Astrometric Errors
------------------

Spatially-varying PSF
---------------------

Dipole Measurement
------------------

Difference Imaging / Image Coaddition
-------------------------------------

Transient Detection
-------------------