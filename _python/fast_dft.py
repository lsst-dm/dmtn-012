"""
A fast DFT approximation for arrays of amplitudes at floating-point locations.

Returns a regularly spaced 2D array with the discrete Fourier transform of all the points.

Points are gridded in image space by evaluating the sinc function, with no folding.

If kernel_radius is set, the sinc function for each point is only evaluated for pixels within a
radius of kernel_radius in pixels, and pixels within slices kernel_radius x M and N x kernel_radius
for an MxN image.

If amplitudes is a two dimensional array (m, n), it is interpreted as a 1D array of m points,
each with n different amplitude values. In this case, a n element list of MxN images will be returned.

Important note:
"""
from __future__ import division
import numpy as np
import cPickle
import unittest
import lsst.utils.tests as utilsTests


def fast_dft(amplitudes, x_loc, y_loc, x_size=None, y_size=None, no_fft=True, kernel_radius=10, **kwargs):
    """
    !Construct a gridded 2D Fourier transform of an array of amplitudes at floating-point locations.
    x_loc - 1D floating point array of x pixel coordinates.
    y_loc - 1D floating point array of y pixel coordinates.
    x_size - desired number of pixels N for the output (M, N) image.
    y_size - desired number of pixels M for the output (M, N) image.
    no_fft - if True, returns the sinc-interpolated image, otherwise takes the FFT [default: True]
    kernel_radius - number of pixels to either side of each source to include [default: 10]
    Accepts **kwargs so that these parameters can be passed through wrappers.
    """
    pi = np.pi

    amplitudes = input_type_check(amplitudes)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)
    if amplitudes.ndim > 1:
        n_cat = amplitudes.shape[1]
        multi_catalog = True
    else:
        n_cat = 1
        multi_catalog = False

    if y_size is None:
        y_size = x_size

    # If the kernel radius is large, it is faster and more accurate to use all of the pixels.
    if pi * kernel_radius**2.0 >= x_size * y_size / 4.0:
        full_flag = True
    else:
        full_flag = False

    def kernel_1d(locs, size):
        """pre-compute the 1D sinc function values along each axis."""
        pix = np.arange(size, dtype=np.float64)
        sign = np.power(-1.0, pix)
        offset = np.floor(locs)
        delta = locs - offset
        kernel = np.zeros((len(locs), size), dtype=np.float64)
        for i, loc in enumerate(locs):
            if delta[i] == 0:
                kernel[i, :][offset[i]] = 1.0
            else:
                kernel[i, :] = np.sin(-pi * loc) / (pi * (pix - loc)) * sign
        return kernel

    def kernel_1d_gen(locs, size):
        """A generalized generator function that pre-computes the 1D sinc function values along one axis."""
        pix = np.arange(size, dtype=np.float64)
        sign = np.power(-1.0, pix)
        for loc in locs:
            offset = np.floor(loc)
            delta = loc - offset
            if delta == 0:
                kernel = np.zeros(size, dtype=np.float64)
                kernel[offset] = 1.0
            else:
                kernel = np.sin(-pi * loc) / (pi * (pix - loc))
                kernel *= sign
            yield kernel
    kernel_x_gen = kernel_1d_gen(x_loc, x_size)
    kernel_y_gen = kernel_1d_gen(y_loc, y_size)

    if multi_catalog:

        def kernel_circle_inds(x_loc, y_loc, kernel_radius=None):
            """
            A generator function that pre-computes the pixels to use for gridding.
            Returns the x and y indices for all pixels within a given radius of a location,
            that are NOT included in slices centered on that location.
            Also applies a Hanning window function for those values, to reduce ringing at the edges.
            """
            ind_radius = int(4 * kernel_radius)
            x_i0, y_i0 = np.meshgrid(np.arange(2.0 * ind_radius), np.arange(2.0 * ind_radius))
            x_i0_int, y_i0_int = np.meshgrid(np.arange(2 * ind_radius), np.arange(2 * ind_radius))
            x_pix_arr = np.round(x_loc)
            y_pix_arr = np.round(y_loc)
            taper_filter = np.hanning(2 * ind_radius)
            taper_filter /= taper_filter[ind_radius - kernel_radius]
            for src_i in range(len(x_loc)):
                x_pix = int(x_pix_arr[src_i])
                y_pix = int(y_pix_arr[src_i])
                dx = x_loc[src_i] - x_pix + ind_radius
                dy = y_loc[src_i] - y_pix + ind_radius

                dist = np.sqrt((x_i0 - dx)**2.0 + (y_i0 - dy)**2.0)
                dist[ind_radius - kernel_radius: ind_radius + kernel_radius, :] = ind_radius
                dist[:, ind_radius - kernel_radius: ind_radius + kernel_radius] = ind_radius
                if x_pix < ind_radius:
                    dist[:, 0: ind_radius - x_pix] = ind_radius
                if x_pix > x_size - ind_radius:
                    dist[:, x_size - ind_radius - x_pix:] = ind_radius
                if y_pix < ind_radius:
                    dist[0: ind_radius - y_pix, :] = ind_radius
                if y_pix > y_size - ind_radius:
                    dist[y_size - ind_radius - y_pix:, :] = ind_radius
                dist_test = dist < ind_radius
                x_i = x_i0_int[dist_test]
                y_i = y_i0_int[dist_test]
                # y_i, x_i = np.where(test_image < ind_radius)
                taper = taper_filter[y_i] * taper_filter[x_i]
                x_i += x_pix - ind_radius
                y_i += y_pix - ind_radius
                yield x_i
                yield y_i
                yield taper

        if full_flag:
            model_img = []
            amp_arr = [amplitudes[:, _i] for _i in range(n_cat)]
            kernel_x = kernel_1d(x_loc, x_size)
            kernel_y = kernel_1d(y_loc, y_size)
            for amp in amp_arr:
                amp_vals = np.einsum('i,ij->ij', amp, kernel_y)
                model_img += [np.einsum('ij,ik->jk', amp_vals, kernel_x)]
        else:
            amp_arr = [amplitudes[_i, :] for _i in range(len(x_loc))]
            model_arr = np.zeros((y_size, x_size, n_cat))
            model_arr_T = np.zeros((x_size, y_size, n_cat))
            x_pix = (int(np.round(xv)) for xv in x_loc)
            y_pix = (int(np.round(yv)) for yv in y_loc)
            kernel_ind_gen = kernel_circle_inds(x_loc, y_loc, kernel_radius=kernel_radius)
            for amp in amp_arr:
                x_c = next(x_pix)
                y_c = next(y_pix)
                x0 = x_c - kernel_radius
                if x0 < 0:
                    x0 = 0
                x1 = x_c + kernel_radius
                if x1 > x_size:
                    x1 = x_size
                y0 = y_c - kernel_radius
                if y0 < 0:
                    y0 = 0
                y1 = y_c + kernel_radius
                if y1 > y_size:
                    y1 = y_size
                kernel_x = next(kernel_x_gen)
                kernel_y = next(kernel_y_gen)
                kernel_x_slice = np.einsum('i,j->ij', kernel_x[x0:x1], kernel_y)
                # central pixels will be added in both slices, so set to zero in one of them
                kernel_x_slice[:, y0:y1] = 0
                kernel_y_slice = np.einsum('i,j->ij', kernel_y[y0:y1], kernel_x)
                # kernel_y_slice[:, x0:x1] /= 2.0
                x_i = next(kernel_ind_gen)
                y_i = next(kernel_ind_gen)
                taper = next(kernel_ind_gen)
                if len(y_i) > 0:
                    shell_vals = np.einsum('i,j->ij', kernel_x[x_i] * kernel_y[y_i] * taper, amp)
                    model_arr[y_i, x_i, :] += shell_vals
                y_vals = np.einsum('ij,k->ijk', kernel_y_slice, amp)
                model_arr[y0:y1, :, :] += y_vals
                x_vals = np.einsum('ij,k->ijk', kernel_x_slice, amp)
                model_arr_T[x0:x1, :, :] += x_vals

            model_arr += np.transpose(model_arr_T, (1, 0, 2))
            model_img = [model_arr[:, :, c_i] for c_i in range(n_cat)]
    else:
        # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D

        kernel_x = kernel_1d(x_loc, x_size)
        kernel_y = (amplitudes * kernel_1d(y_loc, y_size).T).T

        model_img = np.einsum('ij,ik->jk', kernel_y, kernel_x)

    if not no_fft:
        if multi_catalog:
            for _i, model in enumerate(model_img):
                model_img[_i] = np.fft.rfft2(model)
        else:
            model_img = np.fft.rfft2(model_img)
    return(model_img)


def input_type_check(var):
    """Helper function to ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class SingleSourceTestCase(utilsTests.TestCase):
    """Lightweight unit test cases using a single star and a small image."""

    def setUp(self):
        """Define parameters used by every test."""
        self.x_size = 64
        self.y_size = 64
        self.x_loc = [13.34473]  # Arbitrary
        self.y_loc = [42.87311]  # Arbitrary
        self.radius = 10
        n_star = 1
        n_band = 3
        flux_arr = np.zeros((n_star, n_band))
        flux_arr[0, :] = np.arange(n_band) / 10.0 + 1.0
        self.amplitudes = flux_arr

    def tearDown(self):
        """Clean up."""
        del self.x_size
        del self.y_size
        del self.x_loc
        del self.y_loc
        del self.radius
        del self.amplitudes

    def test_single_source(self):
        """Test a single star with a single wavelength slice."""
        data_file = "test_data/SingleSourceTest.npy"
        ref_image = np.load(data_file)
        amplitude = self.amplitudes[0, 0]
        single_image = fast_dft(amplitude, self.x_loc, self.y_loc,
                                x_size=self.x_size, y_size=self.y_size, kernel_radius=self.radius)
        abs_diff_sum = np.sum(np.abs(single_image - ref_image))
        self.assertAlmostEqual(abs_diff_sum, 0.0)

    def test_faint_source(self):
        """Test a single faint star with multiple wavelength slices, using a reduced kernel."""
        data_file = "test_data/FaintSourceTest.npy"
        ref_image = np.load(data_file)
        faint_image = fast_dft(self.amplitudes, self.x_loc, self.y_loc,
                               x_size=self.x_size, y_size=self.y_size, kernel_radius=self.radius)
        abs_diff_sum = 0.0
        for _i, image in enumerate(faint_image):
            abs_diff_sum += np.sum(np.abs(image - ref_image[_i]))
        self.assertAlmostEqual(abs_diff_sum, 0.0)

    def test_bright_source(self):
        """Test a single bright star with multiple wavelength slices, using all pixels."""
        data_file = "test_data/BrightSourceTest.npy"
        ref_image = np.load(data_file)
        bright_image = fast_dft(self.amplitudes, self.x_loc, self.y_loc,
                                x_size=self.x_size, y_size=self.y_size, kernel_radius=self.x_size)
        abs_diff_sum = 0.0
        for _i, image in enumerate(bright_image):
            abs_diff_sum += np.sum(np.abs(image - ref_image[_i]))
        self.assertAlmostEqual(abs_diff_sum, 0.0)


class MultipleSourceTestCase(utilsTests.TestCase):
    """Larger test case intended for profiling and accurate timing."""

    def setUp(self):
        """Define parameters used by every test."""
        seed = 5
        self.x_size = 512
        self.y_size = 512
        self.radius = 10
        n_star = 100
        n_band = 12
        rand_gen = np.random
        rand_gen.seed(seed)
        self.x_loc = rand_gen.uniform(low=0, high=self.x_size, size=n_star)  # Arbitrary
        self.y_loc = rand_gen.uniform(low=0, high=self.y_size, size=n_star)  # Arbitrary
        flux_arr = np.zeros((n_star, n_band))
        for _i in range(n_star):
            flux_arr[_i, :] = np.abs(np.arange(n_band)**2.0 / 10.0 + rand_gen.normal(scale=10.0))
        self.amplitudes = flux_arr

    def tearDown(self):
        """Clean up."""
        del self.x_size
        del self.y_size
        del self.x_loc
        del self.y_loc
        del self.radius
        del self.amplitudes

    def test_continuum_source(self):
        """Test stars with a single wavelength slice."""
        data_file = "test_data/SingleSourceTestLarge.npy"
        ref_image = np.load(data_file)
        amplitude = self.amplitudes[:, 0]
        single_image = fast_dft(amplitude, self.x_loc, self.y_loc,
                                x_size=self.x_size, y_size=self.y_size, kernel_radius=self.radius)
        abs_diff_sum = np.sum(np.abs(single_image - ref_image))
        self.assertAlmostEqual(abs_diff_sum, 0.0)

    def test_faint_source(self):
        """Test faint stars with multiple wavelength slices, using a reduced kernel."""
        data_file = "test_data/FaintSourceTestLarge.npy"
        ref_image = np.load(data_file)
        faint_image = fast_dft(self.amplitudes, self.x_loc, self.y_loc,
                               x_size=self.x_size, y_size=self.y_size, kernel_radius=self.radius)
        abs_diff_sum = 0.0
        for _i, image in enumerate(faint_image):
            abs_diff_sum += np.sum(np.abs(image - ref_image[_i]))
        self.assertAlmostEqual(abs_diff_sum, 0.0)

    def test_bright_source(self):
        """Test bright stars with multiple wavelength slices, using all pixels."""
        data_file = "test_data/BrightSourceTestLarge.npy"
        ref_image = np.load(data_file)
        bright_image = fast_dft(self.amplitudes, self.x_loc, self.y_loc,
                                x_size=self.x_size, y_size=self.y_size, kernel_radius=self.x_size)
        abs_diff_sum = 0.0
        for _i, image in enumerate(bright_image):
            abs_diff_sum += np.sum(np.abs(image - ref_image[_i]))
        self.assertAlmostEqual(abs_diff_sum, 0.0)


def suite():
    """Return a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SingleSourceTestCase)
    suites += unittest.makeSuite(MultipleSourceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests."""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
