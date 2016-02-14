# pygad README

This module is a light-weight Python module for analysis of Gadget snapshots.
It supports all Gadget formats (in our experience better than [pynbody]).
[pygad][] is inspired and influenced by [pynbody].

---

## Getting started

### Prerequisites

Written entirely in `Python 2.7` and `C++` (the latter only for speed relevant parts), it only requires a running Python 2.7 environment and the following additional quasi-standard modules:

* `numpy`
* `scipy`
* `matplotlib`
* `hdf5` (only required if reading/writing format 3 snapshots)

as well as the free library

* `GSL`

and the `g++` compiler that needs to support the C++11 standard.



### Get and Install pygad

Simply clone and install the git repository:

```
$ git clone https://bitbucket.org/broett/pygad
$ cd pygad
$ wget https://bitbucket.org/broett/pygad/downloads/bc03.tar.gz
$ tar -xzf bc03.tar.gz
$ sudo python setup.py install
```

The third step of downloading the [Bruzual & Charlot (2003) SSP models][BC03] is optional but recommended (since required for standard star plotting routines).

If you have problems or want a more detailed explanation, see the [wiki][WikiInstallation].

### Configure

In order to use the single stellar population models

You probably need to customise [pygad]'s config files for your specific type of snapshots (even HDF5 block names can differ).
This goes a little beyond the scope of a README.
Be referred to the [wiki][WikiConfig].

### Use pygad

For a starter you could try something like the following in iPython:

```
#!python
import pygad
import matplotlib.pyplot as plt
s = pygad.Snap('path/to/snap')
snap, halo = pygad.tools.prepare_zoom(s)
R200, M200 = pygad.analysis.virial_info(snap, extent='5 Mpc')
fig, ax, cbar = pygad.plotting.image(snap.gas)
ax.add_artist(plot.Circle([0,0], R200, facecolor='none', edgecolor='w'))
plt.draw()
```

There is a iPython Notebook called `QuickStart.ipynb` in the [bitbucket downloads][Downloads] that shows some more of the features.
Please, also read the [wiki][WikiHome].

---

## Support, Contact

If you have any problems, ideas, found bugs, or want to contribute in some way, please
contact me:  
[broett@mpa-garching.mpg.de](mailto:broett@mpa-garching.mpg.de)

[pygad]: https://bitbucket.org/broett/pygad
[pynbody]: https://pynbody.github.io
[BC03]: http://www.bruzual.org
[Downloads]: https://bitbucket.org/broett/pygad/downloads
[WikiHome]: https://bitbucket.org/broett/pygad/wiki/Home
[WikiInstallation]: https://bitbucket.org/broett/pygad/wiki/Installation
[WikiConfig]: https://bitbucket.org/broett/pygad/wiki/Configuration
