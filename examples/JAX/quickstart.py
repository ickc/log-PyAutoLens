# %% [markdown]
# From <https://jax.readthedocs.io/en/latest/quickstart.html>.

# %%
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacobian, jacrev, jit, random, vmap

# %%
jax.numpy.arange(10)

# %%
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# %%
x = jnp.arange(5.0)

# %%
y = selu(x)
y

# %%
type(x), type(y)

# %%
key = random.key(1701)
x = random.normal(key, (1_000_000,))

# %%
%timeit selu(x).block_until_ready()

# %%
selu_jit = jit(selu)
_ = selu_jit(x)  # compiles on first call

# %%
%timeit selu_jit(x).block_until_ready()

# %%
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

# %%
x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)

# %%
derivative_fn(x_small)

# %%
grad(jit(grad(jit(grad(sum_logistic)))))(1.0)

# %%
jacobian(jnp.exp)(x_small)

# %%
def hessian(fun):
    return jit(jacfwd(jacrev(fun)))

# %%
hessian(sum_logistic)(x_small)

# %%
jax.hessian(sum_logistic)(x_small)

# %%
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))


def apply_matrix(x):
    return jnp.dot(mat, x)

# %%
def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])

# %%
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()

# %%
@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, mat.T)

# %%
np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    batched_apply_matrix(batched_x),
    atol=1e-4,
    rtol=1e-4,
)

# %%
%timeit batched_apply_matrix(batched_x).block_until_ready()

# %%
@jit
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)

# %%
np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    vmap_batched_apply_matrix(batched_x),
    atol=1e-4,
    rtol=1e-4,
)

# %%
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()


