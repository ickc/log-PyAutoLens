# %% [markdown]
# # Interferometer Analysis and Likelihood Function
# 
# This notebook demonstrates the implementation of likelihood functions and matrix operations for interferometer data analysis.

# %%
import numpy as np
from os import path

import likelihood_function_funcs

import autolens as al

# %% [markdown]
# ## Mask Definition
# 
# We define the 'real_space_mask' which determines the grid for evaluating the strong lens image. The lens model is evaluated in real space and then mapped to Fourier Space via the NUFFT. This matrix defines the dimensions of certain matrices used in our likelihood function calculations.

# %%
real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=3.0,
)

# %% [markdown]
# ## Interferometer Dataset
# 
# Load an example interferometer dataset for developing the likelihood function.

# %%
dataset_type = "sma"
dataset_path = path.join("../../../packages/dirac_rse_interferometer", "dataset", dataset_type)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

print(dataset.w_tilde)

# %% [markdown]
# ## Mapping Setup
# 
# Set up the mapping matrix which is the starting point for the interferometer code we aim to JAX-ify.

# %%
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

tracer_to_inversion = al.TracerToInversion(tracer=tracer, dataset=dataset)

inversion = tracer_to_inversion.inversion

mapping_matrix = inversion.mapping_matrix
mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]

print(mapping_matrix)

# %% [markdown]
# ## JAX Function Implementations
# 
# ### 1. Data Vector
# Calculate the data vector using the mapping matrix and dirty image.

# %%
dirty_image = dataset.w_tilde.dirty_image
data_vector = likelihood_function_funcs.data_vector_from(
    mapping_matrix=mapping_matrix,
    dirty_image=dirty_image
)

# %% [markdown]
# ### 2. Curvature Matrix
# Calculate the curvature matrix using the w_tilde matrix.

# %%
w_tilde = dataset.w_tilde.w_matrix
curvature_matrix = likelihood_function_funcs.curvature_matrix_via_w_tilde_from(
    w_tilde=w_tilde,
    mapping_matrix=mapping_matrix
)

# %% [markdown]
# ### 3. Regularization Matrix
# Apply regularization to the curvature matrix using constant regularization.

# %%
regularization_matrix = likelihood_function_funcs.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.source_plane_mesh_grid.neighbors,
    neighbors_sizes=mapper.source_plane_mesh_grid.neighbors.sizes,
)

# %% [markdown]
# ### 4-5. Linear System Solution
# Solve the linear system by combining the curvature and regularization matrices.

# %%
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = likelihood_function_funcs.reconstruction_positive_negative_from(
    curvature_reg_matrix=curvature_reg_matrix,
    data_vector=data_vector
)

# %% [markdown]
# ### 6. Chi-Squared Calculation
# Calculate the chi-squared term using the w_tilde matrix formalism.

# %%
chi_squared_term_1 = np.linalg.multi_dot([
    mapping_matrix,
    w_tilde,
    mapping_matrix,
])

chi_squared_term_2 = -np.multiply(2.0, np.dot(mapping_matrix, dirty_image))

chi_squared = chi_squared_term_1 + chi_squared_term_2

# %% [markdown]
# ### 7. Regularization Term
# Calculate the regularization term that penalizes solutions with large flux differences.

# %%
regularization_term = np.matmul(
    reconstruction.T,
    np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

# %% [markdown]
# ### 8. Determinant Terms
# Calculate the determinant terms that help balance complexity and fitting.

# %%
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

# %% [markdown]
# ### 9. Noise Normalization
# Calculate the noise normalization term for the likelihood function.

# %%
noise_normalization = likelihood_function_funcs.noise_normalization_complex_from(
    noise_map=dataset.noise_map,
)

# %% [markdown]
# ### 10. Final Log-Likelihood
# Compute the final log-likelihood by combining all terms.

# %%
log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)


