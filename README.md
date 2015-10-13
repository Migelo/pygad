# pygad

pygad aims to be a light-weight Python module for analysis of Gadget snapshots.
It supports all Gadget formats (1, 2, and 3, i.e. HDF5 – with and without info blocks).

Written entirely in Python 2.7, it only requires a running Python environment and the following additional modules:  
`numpy`, `scipy`, `matplotlib`.

---

## Getting started

### Get and install pygad

Simply clone the git repository:

```
$ git clone https://bitbucket.org/broett/pygad
```

Now you can ...

...either `cd` into the git directory and use pygad from within there:

```
$ cd pygad
$ ipython --pylab
>>> import pygad
>>> ...
```

...or make the git repository known to the system (recommended):

```
$ echo "export PYTHONPATH=/path/to/pygad:$PYTHONPATH" >> ~/.bashrc
```

if you use bash; if you use tc-shell it is:

```
echo "setenv PYTHONPATH /path/to/pygad:$PYTHONPATH" >> ~/.tcshrc
```

...or install pygad globally (not recommended, since it copies the current version to the system and later changes are not seen by this global version):

```
$ cd pygad
$ python setup.py install
```

In the latter two cases the pygad library then is available from any directory.

### Use pygad

A quick tour is presented in the iPython Notebook `QuickStart.ipynb` in [downloads][Downloads].

The following code snipped demonstrates how easy it can be to do a plot of the gas colour-coded by temperature (the default) that properly treats the SPH smoothing:

```python
import pygad
import matplotlib.pyplot as plt
snap, halo = pygad.prepare('path/to/snapshot')
R200, M200 = pygad.analysis.virial_info(snap)
fig, ax, cbar = pygad.plotting.image(snap.gas, extent='5 Mpc')
ax.add_artist(plot.Circle([0,0], R200, facecolor='none', edgecolor='w'))
plt.draw()
```

![gas colour-coded by temperature](https://bitbucket.org/broett/pygad/raw/3dcd7a449683ef7a199249d042094730ecea8c8a/images/gas_big_T.png)

For colour-coding by (mass-weighted) metallicity one can go on as follows:

```python
fig, ax, cbar = pygad.plotting.image(snap.gas, extent='5 Mpc', colors='Z', colors_av='mass', clogscale=True, clim=[-4,-1])
ax.add_artist(plot.Circle([0,0], R200, facecolor='none', edgecolor='w'))
plt.draw()
```

![gas colour-coded by temperature](https://bitbucket.org/broett/pygad/raw/3dcd7a449683ef7a199249d042094730ecea8c8a/images/gas_big_Z.png)

And a face-on image of the stellar disc:

```python
fig, ax, cbar = pygad.plotting.image(snap.stars, extent='100 kpc')
```

![gas colour-coded by temperature](https://bitbucket.org/broett/pygad/raw/3dcd7a449683ef7a199249d042094730ecea8c8a/images/stars_faceon.png)


For further documentation of the different modules, functions, and classes of pygad,
please use the integrated help of Python (magic `?` at the end or `help()` in plain Python)!

---

## Support, Contact

If you have any problems, ideas, found bugs, or want to contribute in some way, please
contact me:  
[broett@mpa-garching.mpg.de](mailto:broett@mpa-garching.mpg.de)

[Downloads]: https://bitbucket.org/broett/pygad_old/downloads
