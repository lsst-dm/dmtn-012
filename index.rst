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

:tocdepth: 1

Warning

This technical note is currently a draft!

#########
StarFast - A Fast Simulation Building Tool for Testing Algorithms
#########
(StarSim, SimFast, )

Overview
========
StarFast is a simple but fast simulation tool to generate images for testing algorithms. It has been developed primarily with an aim to quickly generate multiple simple but fairly realistic images affected by Differential Chromatic Refraction (DCR) and astrometric errors, but is hopefully general enough to be used for other cases when we need to test algorithms against simulated data. 

StarFast will generate a simulated catalog of stars (see 'Simulated catalogs' below) and compute the flux in Janskys for each source given the instrument bandpass. For wavelength-dependent effects such as DCR, each band is divided into multiple planes with restricted wavelength intervals which will be stacked after all processing. 

.. _section-headings-sim-cat:

Simulated catalogs
==================
StarFast begins by creating a simulated catalog of stars with properties representative of the local population. Properties are drawn from random distributions for the stellar types O-M in proportion to their local abundance, and each star is set at a random distance from 1-100 light years. Simulated properties include temperature (in degrees K), luminosity (relative to solar), surface gravity (relative to solar), and metallicity (log, relative to solar). These properties are matched to Kurucz model SEDs from GalSim, which are scaled by the luminosity of each star and attenuated by distance and the instrument bandpass throughput to yield fluxes in Janskys. Alternately, a simple blackbody radiation spectrum (with the bandpass) can be used. 

Coordinates are also chosen from a random distribution, with no attempt at simulating realistic clustering. All of the random distributions, including stellar properties and coordinates, can be initialized from a user-supplied seed value, which allows for repeated simulations of the same patch of sky under different conditions. Simulated catalogs may also be returned and saved, so they may be modified by external tools if desired, and those saved catalogs may be supplied in place of generating a new random catalog from a seed.



Implementation
==============


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