# %%
from __future__ import annotations

import numpy as np
from os import path

import jax
import jax.numpy as jnp

from autoarray import numba_util
import autolens as al

jax.config.update("jax_enable_x64", True)

# %%
@numba_util.jit(nopython=True, parallel=True)
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray,
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
    native_index_for_slim_index,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    image_native
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros(image_pixels)

    weight_map_real = visibilities_real / noise_map_real**2.0

    for ip0 in range(image_pixels):
        value = 0.0

        y = grid_radians_slim[ip0, 1]
        x = grid_radians_slim[ip0, 0]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            value += weight_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0
                * np.pi
                * (
                    y * uv_wavelengths[vis_1d_index, 0]
                    + x * uv_wavelengths[vis_1d_index, 1]
                )
            )

        w_tilde_data[ip0] = value

    return w_tilde_data


# %% [markdown]
# ## Example
# 
# We now load an interferometer dataset and input the quantities which use the function above to make
# the w tilde matrix.
# 
# ### Mask
# 
# We define the 'real_space_mask' which defines the grid the image the strong lens is evaluated using.
# 
# Basically, the lens model is evaluated in real space and then mapped to Fourier Space via the NUFFT. This
# matrix therefore defines the dimensions of certain matrices which enter our likelihood function and calculation.

# %%
real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=3.0,
)

# %% [markdown]
# ### Interferometer Dataset
# 
# We load an example interferometer dataset which will be used to help us develop the likelihood function.

# %%
dataset_type = "sma"
dataset_path = path.join("..", "..", "packages", "dirac_rse_interferometer", "dataset", dataset_type)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

# %% [markdown]
# ### W Tilde
# 
# The code below calls the w_tilde function above to create the w_tilde matrix, which we need to speed up via JAX.
# 
# Note that we have to put `np.array()` in front of all objects.
# 
# This is because JAX arrays are typed (e.g. `Array2D`) and need special behaviour to ensure JAX worked on the JAX'ed array. This is what `np.array()` achieves.
# 
# The source code now requires these `np.array()`'s to be in place even though we're currently using numba.

# %%
visibilities_real = np.array(dataset.data.real)
noise_map_real = np.array(dataset.noise_map.real)
uv_wavelengths = np.array(dataset.uv_wavelengths)
grid_radians_slim = np.array(dataset.grid.in_radians)
native_index_for_slim_index = np.array(real_space_mask.derive_indexes.native_for_slim).astype("int")

# %%
w_tilde_ref = w_tilde_data_interferometer_from(
    visibilities_real=visibilities_real,
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
    native_index_for_slim_index=native_index_for_slim_index,
)

# %%
import numba


@numba.jit("float64[::1](float64[::1], float64[::1], float64[:,::1], float64[:,::1], int64[:,::1])", nopython=True, nogil=True, parallel=True)
def w_tilde_data_interferometer_from_numba(
    visibilities_real: np.ndarray[tuple[int], np.float64],
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    .. math::
        \tilde{w}_{\text{data},i} = \sum_{j=1}^N \left(\frac{N_{r,j}^2}{V_{r,j}}\right)^2 \cos\left(2\pi(g_{i,1}u_{j,0} + g_{i,0}u_{j,1})\right)

    Parameters
    ----------
    visibilities_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray, shape (M,), dtype=float64
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    g_i = grid_radians_slim.reshape(-1, 1, 2)
    u_j = uv_wavelengths.reshape(1, -1, 2)
    return (
        # (1, j∊N)
        np.square(np.square(noise_map_real) / visibilities_real).reshape(1, -1) *
        np.cos(
            (2.0 * np.pi) *
            # (i∊M, j∊N)
            (
                g_i[:, :, 0] * u_j[:, :, 1] +
                g_i[:, :, 1] * u_j[:, :, 0]
            )
        )
    ).sum(axis=1)  # sum over j

# %%
@jax.jit
def w_tilde_data_interferometer_from_jax(
    visibilities_real: np.ndarray[tuple[int], np.float64],
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
    native_index_for_slim_index: np.ndarray[tuple[int, int], np.int64],
) -> np.ndarray[tuple[int], np.float64]:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    .. math::
        \tilde{w}_{\text{data},i} = \sum_{j=1}^N \left(\frac{N_{r,j}^2}{V_{r,j}}\right)^2 \cos\left(2\pi(g_{i,1}u_{j,0} + g_{i,0}u_{j,1})\right)

    Parameters
    ----------
    visibilities_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_real : ndarray, shape (N,), dtype=float64
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
    native_index_for_slim_index : ndarray, shape (M, 2), dtype=int64
        An array that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray, shape (M,), dtype=float64
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """
    g_i = grid_radians_slim.reshape(-1, 1, 2)
    u_j = uv_wavelengths.reshape(1, -1, 2)
    return (
        # (1, j∊N)
        jnp.square(jnp.square(noise_map_real) / visibilities_real).reshape(1, -1) *
        jnp.cos(
            (2.0 * jnp.pi) *
            # (i∊M, j∊N)
            (
                g_i[:, :, 0] * u_j[:, :, 1] +
                g_i[:, :, 1] * u_j[:, :, 0]
            )
        )
    ).sum(axis=1)  # sum over j

# %%
w_tilde_numba = w_tilde_data_interferometer_from_numba(
    visibilities_real=visibilities_real,
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
    native_index_for_slim_index=native_index_for_slim_index,
)

# %%
np.testing.assert_allclose(w_tilde_numba, w_tilde_ref)

# %%
w_tilde_jax = w_tilde_data_interferometer_from_jax(
    visibilities_real=visibilities_real,
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
    native_index_for_slim_index=native_index_for_slim_index,
)

# %%
np.testing.assert_allclose(w_tilde_jax, w_tilde_ref)



# %%
N = 1000
M = 1000
visibilities_real = np.random.normal(size=N)
noise_map_real = np.random.normal(size=N)
uv_wavelengths = np.random.normal(size=(N, 2))
grid_radians_slim = np.random.normal(size=(M, 2))
native_index_for_slim_index = np.random.randint(0, M, size=(M, 2))

# %%
%timeit w_tilde_data_interferometer_from(visibilities_real, noise_map_real, uv_wavelengths, grid_radians_slim, native_index_for_slim_index)

# %%
%timeit w_tilde_data_interferometer_from_numba(visibilities_real, noise_map_real, uv_wavelengths, grid_radians_slim, native_index_for_slim_index)

# %%
%timeit w_tilde_data_interferometer_from_jax(visibilities_real, noise_map_real, uv_wavelengths, grid_radians_slim, native_index_for_slim_index).block_until_ready()
# %%
