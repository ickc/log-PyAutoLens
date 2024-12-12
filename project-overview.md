From <https://docs.google.com/document/d/1jGlxdnpjX5t4rXKBc0hrRx9vWcgXXnoR_5RY2nWTeUs/edit?usp=sharing>

# Project Structure & Overview

**PyAutoLens** is found at the following GitHub repository: [https://github.com/Jammy2211/PyAutoLens](https://github.com/Jammy2211/PyAutoLens)

However, a large fraction of functionality used by PyAutoLens (which we will need to JAX-ify) is actually contained in the following 4 parent projects: 

- **PyAutoConf:** [https://github.com/rhayes777/PyAutoConf](https://github.com/rhayes777/PyAutoConf) hands config files, probably not relevant for JAX development.

- **PyAutoFit:** [https://github.com/rhayes777/PyAutoFit](https://github.com/rhayes777/PyAutoFit) (Statistics library which interfaces with JAX when fitting models).

- **PyAutoArray:** [https://github.com/Jammy2211/PyAutoArray](https://github.com/Jammy2211/PyAutoArray) (Extends NumPy array with custom data structures for performing calculations, including functionality which can use JAX arrays instead of numpy arrays).

- **PyAutoGalaxy:** [https://github.com/Jammy2211/PyAutoGalaxy](https://github.com/Jammy2211/PyAutoGalaxy) (Fits non-lensed galaxies, will be where we start JAX development).

You will want to start by **git clone** all these repos – Instructions of setting this up are here: [https://pyautolens.readthedocs.io/en/latest/installation/source.html](https://pyautolens.readthedocs.io/en/latest/installation/source.html)

## Workspaces

The above repos are the **source code repositories**, where development takes place and which are released to PyPI.

After a user has installed PyAutoLens they download a workspace which contains examples and tutorial.

To simplify JAX development, we will start working with **PyAutoGalaxy** and therefore should check out the **autogalaxy_workspace:** [https://github.com/Jammy2211/autogalaxy_workspace](https://github.com/Jammy2211/autogalaxy_workspace/tree/release)

**Make sure you clone the `main` branch, not the `release` branch, noting that main is not the default branch!!!**

## Initial Scripts

Following scripts are worth a read through to get a sense of the scientific scope of the project:
- [https://github.com/Jammy2211/autogalaxy_workspace/blob/release/introduction.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/introduction.ipynb)
- [https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_1_galaxies.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_1_galaxies.ipynb)
- [https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_2_fit.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_2_fit.ipynb)
- [https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_3_modeling.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_3_modeling.ipynb)

## Guides

This is the step by step guide to the likelihood function we will speed up with JAX:  
[https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric.ipynb)

The following contributor_guide explains where code is, however this covers more functionality than we will start with so I am going to update this to only include what you need to get started:

[https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric/contributor_guide.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric/contributor_guide.ipynb)

## PyAutoArray Data Structures

The project PyAutoArray creates a number of Python classes which extend np.ndarray with functionality required for the astronomy projects:

- [https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/structures](https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/structures)
- [https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/arrays/uniform_2d.py](https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/arrays/uniform_2d.py)
- [https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/uniform_2d.py](https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/uniform_2d.py)

This initially caused problems with JAX, as JAX requires one to use JAX arrays instead of numpy arrays, so a lot of conflicts arose.  
Rich (who was in the Zoom call) implemented a lot of functionality that meant the source code provided simultaneous support for numpy arrays (including numba calculations) and JAX arrays:

- [https://github.com/Jammy2211/PyAutoArray/pull/80](https://github.com/Jammy2211/PyAutoArray/pull/80)
- [https://github.com/Jammy2211/PyAutoGalaxy/pull/147](https://github.com/Jammy2211/PyAutoGalaxy/pull/147)

This is why many of these data structures have weird Python functions like this:  
[https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/abstract_ndarray.py](https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/abstract_ndarray.py)

```python
@classmethod
def instance_unflatten(cls, aux_data, children):
    """
    Unflatten a tuple of attributes (i.e. a pytree) into an instance of an autoarray class
    """
    instance = cls.__new__(cls)
    for key, value in zip(aux_data, children[1:]):
        setattr(instance, key, value)

    return instance

def __copy__(self):
    """
    When copying an autoarray also copy its underlying array.
    """
    new = self.__new__(self.__class__)
    new.__dict__.update(self.__dict__)
    new._array = self._array.copy()
    return new

def __deepcopy__(self, memo):
    """
    When copying an autoarray also copy its underlying array.
    """
    new = self.__new__(self.__class__)
    new.__dict__.update(self.__dict__)
    new._array = self._array.copy()

    return new
```

We will probably run into issues with these data structures throughout the project, but Rich is available to resolve them for us if we hit any road blocks.

A good starting point is getting your opinion on whether this is a sane way to handle things long-term.

## JAX Support

The PR's above therefore mean that the source code already supports JAX, in so far as if JAX is installed in your virtual environment (pip install jax and pip install jaxlib) then JAX arrays are used throughout calculations instead of ndarrays.

I just installed JAX and ran the unit tests, and noted that some of the **PyAutoGalaxy** unit tests fail. JAX support works fine in general on the source code.

In order for JAX functionality to work, you need to not only install jax (pip install jax and pip install jaxlib) but also run the Python code with the enviroment variable USE_JAX=1 (export USE_JAX=1).

To run the scripts below, use the following JAX branches:

- [https://github.com/rhayes777/PyAutoFit/tree/feature/jax_assert_disable](https://github.com/rhayes777/PyAutoFit/tree/feature/jax_assert_disable)
- [https://github.com/Jammy2211/PyAutoArray/tree/feature/jax_wrapper](https://github.com/Jammy2211/PyAutoArray/tree/feature/jax_wrapper)
- [https://github.com/Jammy2211/PyAutoGalaxy/tree/feature/jax_wrapper](https://github.com/Jammy2211/PyAutoGalaxy/tree/feature/jax_wrapper)

## JAX Test

We have test repositories for testing and development, so you should go ahead and clone the autogalaxy_workspace_test: [https://github.com/Jammy2211/autogalaxy_workspace_test](https://github.com/Jammy2211/autogalaxy_workspace_test)

This test workspace has the first example script we will use to start testing JAX calls and GPU run times:  
[https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad.py](https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad.py)

The JAX test above goes via the source code, which at first is a bit of a pain to navigate. I have therefore written a self contained piece of code, whereby I extracted the source function calls and wrote them sequentially as a simpler starting point: 

[https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad_manual.py](https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad_manual.py)

**This currently raises an exception… which I am unsure how to fix, so fixing this is probably our first goal!**

Basically, this script calls the likelihood function describes in the step-by-step guide here ([https://github.com/Jammy2211/autogalaxy_workspace/tree/main/notebooks/imaging/advanced/log_likelihood_function/parametric](https://github.com/Jammy2211/autogalaxy_workspace/tree/main/notebooks/imaging/advanced/log_likelihood_function/parametric)) but puts it through a JAX grad function, which does two things:

## Next Steps

Complete a simple conversion of numba code to JAX code described here:

[https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax_examples/task_2_simple_conversions/func_grad_manual.py](https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax_examples/task_2_simple_conversions/func_grad_manual.py)

## Useful Modules

- [https://github.com/rhayes777/PyAutoFit/blob/feature/jax_wrapper/autofit/jax_wrapper.py](https://github.com/rhayes777/PyAutoFit/blob/feature/jax_assert_disable/autofit/jax_wrapper.py)
- [https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/numpy_wrapper.py](https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/numpy_wrapper.py)
