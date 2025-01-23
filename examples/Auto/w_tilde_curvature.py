# %%
import numpy as np
from os import path

from autoarray import numba_util

import autolens as al

from jax import jit
import jax.numpy as jnp
import jax

import h5py

jax.config.update("jax_enable_x64", True)

# %%
@numba_util.jit(nopython=True, parallel=True)
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde

# %% [markdown]
# # Example
# 
# We now load an interferometer dataset and input the quantities which use the function above to make
# the w tilde matrix.
# 
# ## Mask
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
# ## Interferometer Dataset
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
# ## W Tilde
# 
# The code below calls the w_tilde function above to create the w_tilde matrix, which we need to speed up via JAX.
# 
# Note that we have to put `np.array()` in front of all objects to ensure JAX compatibility.

# %%
noise_map_real = np.array(dataset.noise_map.real)
uv_wavelengths = np.array(dataset.uv_wavelengths)
grid_radians_slim = np.array(dataset.grid.in_radians)

w_tilde_ref = w_tilde_curvature_interferometer_from(
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
)

# %%
for name in ("noise_map_real", "uv_wavelengths", "grid_radians_slim", "w_tilde_ref"):
    obj = locals()[name]
    print(f"===== {name} =====")
    np.info(obj)

# %%
import numba


@numba.jit("f8[:, ::1](f8[::1], f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, parallel=True)
def w_tilde_curvature_interferometer_from_numba(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    .. math::
        W̃_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    Parameters
    ----------
    noise_map_real : ndarray, shape (N,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """
    # (i∊M, j∊M, 1, 2)
    g_ij =  grid_radians_slim.reshape(-1, 1, 1, 2) - grid_radians_slim.reshape(1, -1, 1, 2)
    # (1, 1, k∊N, 2)
    u_k = uv_wavelengths.reshape(1, 1, -1, 2)
    return (
        np.cos(
            (2.0 * np.pi) *
            # (M, M, N)
            (
                g_ij[:, :, :, 0] * u_k[:, :, :, 1] +
                g_ij[:, :, :, 1] * u_k[:, :, :, 0]
            )
        ) /
        # (1, 1, k∊N)
        np.square(noise_map_real).reshape(1, 1, -1)
    ).sum(2)  # sum over k

# %%
@jit
def w_tilde_curvature_interferometer_from_jax(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    .. math::
        W̃_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    Parameters
    ----------
    noise_map_real : ndarray, shape (N,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (N, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """
    # (i∊M, j∊M, 1, 2)
    g_ij =  grid_radians_slim.reshape(-1, 1, 1, 2) - grid_radians_slim.reshape(1, -1, 1, 2)
    # (1, 1, k∊N, 2)
    u_k = uv_wavelengths.reshape(1, 1, -1, 2)
    return (
        jnp.cos(
            (2.0 * jnp.pi) *
            # (M, M, N)
            (
                g_ij[:, :, :, 0] * u_k[:, :, :, 1] +
                g_ij[:, :, :, 1] * u_k[:, :, :, 0]
            )
        ) /
        # (1, 1, k∊N)
        jnp.square(noise_map_real).reshape(1, 1, -1)
    ).sum(2)  # sum over k

# %%
w_tilde_numba = w_tilde_curvature_interferometer_from_numba(
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
)

# %%
w_tilde_jax = w_tilde_curvature_interferometer_from_jax(
    noise_map_real=noise_map_real,
    uv_wavelengths=uv_wavelengths,
    grid_radians_slim=grid_radians_slim,
)

# %%
np.testing.assert_allclose(w_tilde_numba, w_tilde_ref)

# %%
np.testing.assert_allclose(w_tilde_jax, w_tilde_ref)

# %%
N = 1000
M = 1000
noise_map_real = np.random.rand(N)
uv_wavelengths = np.random.rand(N, 2)
grid_radians_slim = np.random.rand(M, 2)

# %%
%timeit w_tilde_curvature_interferometer_from(noise_map_real, uv_wavelengths, grid_radians_slim)

# %%
%timeit w_tilde_curvature_interferometer_from_numba(noise_map_real, uv_wavelengths, grid_radians_slim)

# %%
%timeit w_tilde_curvature_interferometer_from_jax(noise_map_real, uv_wavelengths, grid_radians_slim).block_until_ready()
