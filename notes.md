# Overview

- DiRAC: revealing the nature of dark matter with the James Webb space telescope and JAX
- aim: add JAX support in PyAutoLens
- Steven is working on this project
- end in Feb 21st 2025
- join
    - Slack
    - bi-weekly meeting on the science of lensing, occasionally jax
    - DIRAC cluster?
- people:
    - James (Durham)
    - Gokmen
    - Coleman is the one to ask about jax
- schedule a weekly meeting in the week of 12/9
- learn jax

# Questions

- these are deleted:
    https://github.com/Jammy2211/autogalaxy_workspace/blob/release/introduction.ipynb
    
    https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_1_galaxies.ipynb
    
    https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_2_fit.ipynb
    
    https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/overview/overview_3_modeling.ipynb

    https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric.ipynb

    https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/parametric/contributor_guide.ipynb

    https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad.py

    https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax/func_grad_manual.py

    https://github.com/Jammy2211/autogalaxy_workspace/tree/main/notebooks/imaging/advanced/log_likelihood_function/parametric

- "but puts it through a JAX grad function, which does two things"... then?

- Was the note written for someone else earlier? What has changed since then?

- Why wrappers?

- What is "workspace" as in `autogalaxy_workspace_test` or `autogalaxy_workspace`?

TODO:

- [ ] Go through <https://github.com/Jammy2211/autogalaxy_workspace_test/blob/master/jax_examples/task_2_simple_conversions/func_grad_manual.py>

# PyAutoLens Intro

From @Jam:

> Busy day so will send expanded instructions later but this is the Google Doc which explains the project layout and how to get started:[https://docs.google.com/document/d/1jGlxdnpjX5t4rXKBc0hrRx9vWcgXXnoR_5RY2nWTeUs/edit?usp=sharing](https://docs.google.com/document/d/1jGlxdnpjX5t4rXKBc0hrRx9vWcgXXnoR_5RY2nWTeUs/edit?usp=sharing)The project is spread out over multiple repos so takes a bit of getting used to.This is the simple task which is probably a good starting point as an example of trying to convert code from numba to JAX:[https://github.com/Jammy2211/autogalaxy_workspace_test/tree/master/jax_examples/task_2_simple_conversions](https://github.com/Jammy2211/autogalaxy_workspace_test/tree/master/jax_examples/task_2_simple_conversions)

## Environment

I created a reproducible-ish environment using pixi.

Status:

- all `PyAuto*`'s dependencies added.

    > WARN These conda-packages will be overridden by pypi:
    >        networkx, anesthetic, jsonpickle

- [ ] setup `PyAuto*` as submodules and use editable dependencies documented in <https://pixi.sh/latest/reference/pixi_manifest/#pypi-dependencies>.

- [ ] run tests

## Issues

From @Jam:

> The way I approach this is the project has a depedency hierarcrchy of PyAutoFit -> PyAutoArray -> PyAutoGalaxy -> PyAutoLens, so post it in the highest project the issue is relevent.

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

### Git submodules

```bash
git submodule add -b main https://github.com/Jammy2211/autogalaxy_workspace.git packages/autogalaxy_workspace
git submodule add -b main https://github.com/Jammy2211/PyAutoArray.git packages/PyAutoArray
git submodule add -b main https://github.com/Jammy2211/PyAutoGalaxy.git packages/PyAutoGalaxy
git submodule add -b main https://github.com/Jammy2211/PyAutoLens.git packages/PyAutoLens
git submodule add -b main https://github.com/rhayes777/PyAutoConf.git packages/PyAutoConf
git submodule add -b main https://github.com/rhayes777/PyAutoFit.git packages/PyAutoFit
git submodule add -b master https://github.com/Jammy2211/autogalaxy_workspace_test.git packages/autogalaxy_workspace_test
```
