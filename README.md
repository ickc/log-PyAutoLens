A logbook for the project "DiRAC: revealing the nature of dark matter with the James Webb space telescope and JAX".

This should contains all the notes about the project, notebooks and scripts I experimented with, environment(s) I setup for the project, which includes packages related to this project as submodules

Contribution worthy contents should not resides here, but as PRs to respective repos.

# Project overview

- DiRAC: revealing the nature of dark matter with the James Webb space telescope and JAX
- aim: add JAX support in PyAutoLens
- Steven is working on this project
- end in Feb 21st 2025
- join
    - [x] Slack
    - [ ] bi-weekly meeting on the science of lensing, occasionally jax
        - Gokmen will forward the Zoom link to me
    - [x] DIRAC cluster?
- people:
    - James (Durham)
    - Gokmen
    - Coleman is the one to ask about jax
    - Rich developed PyAutoLens from scratch for 6-7 years
- [x] schedule a weekly meeting in the week of 12/9
- [ ] learn jax @inprogress
- Gokmen will ask James about tasks split between us

# Questions

- mentioned 4 notebooks, last 2 repeated:
    * [autogalaxy\_workspace/notebooks/advanced/log\_likelihood\_function/imaging/light\_profile/log\_likelihood\_function.ipynb at release · Jammy2211/autogalaxy\_workspace](https://github.com/Jammy2211/autogalaxy\_workspace/blob/release/notebooks/advanced/log\_likelihood\_function/imaging/light\_profile/log\_likelihood\_function.ipynb)
    * [autogalaxy\_workspace/notebooks/advanced/log\_likelihood\_function/imaging/linear\_light\_profile at release · Jammy2211/autogalaxy\_workspace](https://github.com/Jammy2211/autogalaxy\_workspace/tree/release/notebooks/advanced/log\_likelihood\_function/imaging/linear\_light\_profile)
    * [autogalaxy\_workspace/notebooks/advanced/log\_likelihood\_function/interferometer/light\_profile/log\_likelihood\_function.ipynb at release · Jammy2211/autogalaxy\_workspace](https://github.com/Jammy2211/autogalaxy\_workspace/blob/release/notebooks/advanced/log\_likelihood\_function/interferometer/light\_profile/log\_likelihood\_function.ipynb)

# TODO

- [ ] request projects dp004 and do018 for benchmarking via https://safe.epcc.ed.ac.uk/dirac @wait(for approval)
    - [ ] setup /snap8/scratch/dp004/dc-kili1/RAC16/PyAutoLens
- [x] refactor ported functions per implementation
    - [x] setup unit test
    - [ ] setup doc
    - [x] set up benchmark to compare implementations
        - [x] repeat previous rudimentary benchmark with float64 and double check for consistency
        - [ ] compare results from pytest-benchmark to manually running it. The results seems wildly different
            - perhaps write a manual benchmark with cli and ensure all available threads are used? Perhaps add a matmul in the beginning as a control.
    - [ ] Update duplicated code from <https://github.com/Jammy2211/dirac_rse_interferometer/> to this repo
- [ ] explore calculation of symmetric matrix under JAX framework
- Go through
    - [x] <https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax_examples/task_2_simple_conversions/func_grad_manual.py>
    * [ ] [autogalaxy_workspace/notebooks/advanced/log_likelihood_function/imaging/light_profile/log_likelihood_function.ipynb at release · Jammy2211/autogalaxy_workspace](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/advanced/log_likelihood_function/imaging/light_profile/log_likelihood_function.ipynb)
    * [ ] [autogalaxy_workspace/notebooks/advanced/log_likelihood_function/imaging/linear_light_profile at release · Jammy2211/autogalaxy_workspace](https://github.com/Jammy2211/autogalaxy_workspace/tree/release/notebooks/advanced/log_likelihood_function/imaging/linear_light_profile)
    * [ ] [autogalaxy_workspace/notebooks/advanced/log_likelihood_function/interferometer/light_profile/log_likelihood_function.ipynb at release · Jammy2211/autogalaxy_workspace](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/advanced/log_likelihood_function/interferometer/light_profile/log_likelihood_function.ipynb)
    * [ ] [autogalaxy_workspace/notebooks/advanced/log_likelihood_function/imaging/pixelization/log_likelihood_function.ipynb at release · Jammy2211/autogalaxy_workspace](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/advanced/log_likelihood_function/imaging/pixelization/log_likelihood_function.ipynb)?
- go through <https://github.com/Jammy2211/dirac_rse_interferometer>
    > At the moment, you only care about this script, which shows how a simple dataset we are going to fit is simulated:
    >
    > https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/simulators/sma.py
    >
    > And this script, which illustrates the function we discussed which can currently take 2+ weeks to run:
    >
    > https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/w_tilde.py

    - [x] https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/w_tilde.py

        $$\begin{aligned}
        \mathbf{V}_r &: \text{Real visibilities vector} \\
        \mathbf{N}_r &: \text{Real noise map vector} \\
        \mathbf{u} &: \text{UV wavelengths matrix} \\
        \mathbf{g} &: \text{Grid radians coordinates matrix} \\
        M &: \text{Number of image pixels} \\
        N &: \text{Number of visibility points}
        \end{aligned}$$

        $$\tilde{w}_{\text{data},i} = \sum_{j=1}^N \left(\frac{N_{r,j}^2}{V_{r,j}}\right)^2 \cos\left(2\pi(g_{i,1}u_{j,0} + g_{i,0}u_{j,1})\right)$$

        where $$i \in [0,M-1]$$ and output is vector $$\tilde{\mathbf{w}}_\text{data} \in \mathbb{R}^M$$

    - [x] https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/w_tilde/w_tilde_curvature.py

        $$W̃_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])$$

    - [ ] https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/simulators/sma.py
        - seems to be from https://github.com/Jammy2211/autolens_workspace/blob/main/notebooks/simulators/interferometer/instruments/sma.ipynb

> Another interesting question which could prove a problem with the full JAX implementation  is how to get a 2d delaunay mesh to work, which currently uses scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
> We also use a voronoi mesh with 2d natural neighbour interpolation.
> My understanding is getting these to run in jax could be very hard, but it would be good to scope out if it looks at all feasible.
> https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/mesh/triangulation_2d.py
> triangulation_2d.py
> 
> But the first round of JAX conversions I'll sort the examples for don't need these meshes, that kind of the problem we face after getting the easier to convert code sorted.
> 
> It may also make sense to implement them via this tool https://arxiv.org/abs/2403.08847 first, which will have limitations, and worry about a full jax implementation later

---

More notes from James on Jan 15th 2025:

> In preparation for your return, I've put together a lot of materials to hopefully help you understand the problem better, rather than it just being a bunch of abstract numpy arrays which get mushed together.
>
> I would approach the work via the following steps (you may of done some):
>
> Read this example, which provides a very basic explanation of how a galaxy light profile is used to fit an image of a galaxy. This example uses imaging data, as opposed to interferometer data, because it is conceptually simpler and therefore a better starting point <https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/advanced/log_likelihood_function/imaging/light_profile/log_likelihood_function.ipynb>
>
> 2. Read this example, which again using imaging data (for simplicity) explains how the problem can use linear algebra to solve for the light of the galaxy: <https://github.com/Jammy2211/autogalaxy_workspace/tree/main/notebooks/advanced/log_likelihood_function/imaging/linear_light_profile>
>
> 3. Read this example, which using imaging data explains how a pixelized reconstruction of a galaxy is performed: <https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/advanced/log_likelihood_function/imaging/pixelization/log_likelihood_function.ipynb>
>
> 4. Read this example, which now shows how the simple galaxy light profile calculation is performed for interferometer data, which includes a non-uniform Fast Fourier transform: <https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/advanced/log_likelihood_function/interferometer/light_profile/log_likelihood_function.ipynb>
>
> 5. Read this example, which explains how a pixelized source is performed on interferometer data: <https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/advanced/log_likelihood_function/interferometer/pixelization/log_likelihood_function.ipynb>
>
> 6. Read this example, which explains how the "w-tilde" linear algebra formalism (which you converted some functions to JAX already for) changes the calculation in step 5: <https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/advanced/log_likelihood_function/interferometer/pixelization/w_tilde.ipynb>
>
> 7. Begin to convert all these steps to JAX in the GitHub repo we are working on: <https://github.com/Jammy2211/dirac_rse_interferometer/blob/main/inversion/likelihood_function.py>
>
> By building up a deeper understanding of the problem end-to-end, I think we'll find the overall JAX implementation goes smoother, especially as we will have to slowly extend the code to encompass every step of the likelihood function by the end. However, for now, I think the task is broken down in a way that should be manageable.
>
> I will look at your PR's in the next few days and as I said above, I am working on making the JAX feature branch source code a lot more useable (e.g. refactor, fix unit tests) so we can easily incorporate your JAX'd code outside the source code into the source code without too many headaches.

# PyAutoLens Intro

From @Jam:

> Busy day so will send expanded instructions later but this is the Google Doc which explains the project layout and how to get started:[https://docs.google.com/document/d/1jGlxdnpjX5t4rXKBc0hrRx9vWcgXXnoR_5RY2nWTeUs/edit?usp=sharing](https://docs.google.com/document/d/1jGlxdnpjX5t4rXKBc0hrRx9vWcgXXnoR_5RY2nWTeUs/edit?usp=sharing)The project is spread out over multiple repos so takes a bit of getting used to.This is the simple task which is probably a good starting point as an example of trying to convert code from numba to JAX:[https://github.com/Jammy2211/autogalaxy_workspace_test/tree/master/jax_examples/task_2_simple_conversions](https://github.com/Jammy2211/autogalaxy_workspace_test/tree/master/jax_examples/task_2_simple_conversions)

- branch: feature/jax_wrapper for JAX
- E.g. PR to fix JAX wrapper: https://github.com/Jammy2211/PyAutoArray/pull/156
- [x] Work out of source code, porting, then worry about putting it back later (get initial out of source code repo from @jam)
- [x] get notebooks, papers from @jam

## Environment

I created a reproducible-ish environment using pixi.

Status:

- all `PyAuto*`'s dependencies added.

    > WARN These conda-packages will be overridden by pypi:
    >        networkx, anesthetic, jsonpickle

- [x] setup `PyAuto*` as submodules and use editable dependencies documented in <https://pixi.sh/latest/reference/pixi_manifest/#pypi-dependencies>.

    ```bash
    git submodule add -b main https://github.com/Jammy2211/autogalaxy_workspace.git packages/autogalaxy_workspace
    git submodule add -b main https://github.com/Jammy2211/PyAutoArray.git packages/PyAutoArray
    git submodule add -b main https://github.com/Jammy2211/PyAutoGalaxy.git packages/PyAutoGalaxy
    git submodule add -b main https://github.com/Jammy2211/PyAutoLens.git packages/PyAutoLens
    git submodule add -b main https://github.com/rhayes777/PyAutoConf.git packages/PyAutoConf
    git submodule add -b main https://github.com/rhayes777/PyAutoFit.git packages/PyAutoFit
    git submodule add -b main https://github.com/Jammy2211/dirac_rse_interferometer.git packages/dirac_rse_interferometer
    git submodule add -b master https://github.com/Jammy2211/autogalaxy_workspace_test.git packages/autogalaxy_workspace_test
    git submodule add -b main https://github.com/Jammy2211/autolens_workspace.git packages/autolens_workspace
    git submodule add -b main git@github.com:ickc/python-autojax.git packages/autojax
    ```

- [x] run tests

## Issues

From @Jam:

> The way I approach this is the project has a dependency hierarchy of PyAutoFit -> PyAutoArray -> PyAutoGalaxy -> PyAutoLens, so post it in the highest project the issue is relevant.

### Potential issues

- <https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/numba_util.py>: try and except requires better handlings

# DiRAC COSMA

## Links

* [DiRAC High Performance Computing Facility – Supporting the STFC theory community](https://dirac.ac.uk/)
* [DIRAC Project - hpcicc: Durham](https://safe.epcc.ed.ac.uk/dirac/Project/hpcicc/project\_member.jsp)
* [Memory Intensive Service: Durham – DiRAC High Performance Computing Facility](https://dirac.ac.uk/memory-intensive-durham/)
* [COSMA Facilities — cosma 0.1 documentation](https://cosma.readthedocs.io/en/latest/facilities.html)

## Notes

data (scratch?): `/cosma5/data/durham/dc-cheu2`
apps: `/cosma/apps/durham/dc-cheu2`
compute: only have access to the COSMA5 system:

```bash
salloc --partition=cosma --time=02:00:00 --nodes=1 -A durham
salloc --partition=cosma5 --time=02:00:00 --nodes=1 -A durham
```

> If you are a DiRAC user, please use login7 or [login8.cosma.dur.ac.uk](login8.cosma.dur.ac.uk),  
and submit to the cosma7 or cosma8 queues as appropriate for your  
project.

> If you are not a DiRAC user, please use [login5.cosma.dur.ac.uk](login5.cosma.dur.ac.uk) and  
submit to the cosma queue.

> The quota command will show you your disk quotas and current usage on  
the various different file systems.

> All file systems except for /cosma/home and /cosma/apps are Lustre  
file systems.

TODO: use XDG env var to enforce the following instead of relying on root's symlink:

```
lrwxrwxrwx 1 root root 38 Dec  3 09:44 ./.apptainer -> /cosma/apps/durham/dc-cheu2/.apptainer
lrwxrwxrwx 1 root root 34 Dec  3 09:44 ./.cache -> /cosma/apps/durham/dc-cheu2/.cache
lrwxrwxrwx 1 root root 34 Dec  3 09:44 ./.conda -> /cosma/apps/durham/dc-cheu2/.conda
lrwxrwxrwx 1 root root 35 Dec  3 09:44 ./.config -> /cosma/apps/durham/dc-cheu2/.config
lrwxrwxrwx 1 root root 34 Dec  3 09:44 ./.local -> /cosma/apps/durham/dc-cheu2/.local
lrwxrwxrwx 1 root root 34 Dec  3 09:44 ./.spack -> /cosma/apps/durham/dc-cheu2/.spack
```

Which can be reproduce by:

```bash
TARGET_DIR="/cosma/apps/durham/dc-cheu2"
LINKS=(.apptainer .cache .conda .local .spack)
for LINK in "${LINKS[@]}"; do
    ln -s "$TARGET_DIR/$LINK" "$LINK"
done
```

# References

> I have not read all that much so can't recommend a book but have seen this github recommended: [ratt-ru/foi-course: Fundamentals of Radio Interferometry and Aperture Synthesis Book](https://github.com/ratt-ru/foi-course)
