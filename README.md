# pygad README

This module is a light-weighted (despite having ~25.000 lines of code and documentation) but comprehensive Python module that serves as a framework and basis for analysis of Gadget snapshots.

It supports all Gadget formats and is inspired and influenced by [pynbody].
[pygad] can read in Rockstar output, plot maps of any quantity, generate mock
absorption spectra, and much more.
However, its main power is its framework to conveniently analyse gadget snapshots
without the need to worry about format, read-in, indexing of blocks, units, etc.

---

## Getting started

### Prerequisites

Written entirely in `Python 3.x` and `C++` (the latter only for speed relevant parts), it only requires a running Python 3.x environment and the following additional quasi-standard modules:

* `numpy`
* `scipy`
* `matplotlib`
* `astropy`
* `Pillow`
* `h5py` (only required if reading/writing format 3 snapshots)

as well as the free library

* [`GSL`][GSL]

and the `g++` compiler that needs to support the C++11 standard (for GCC, this is
for version >=4.7).



### Get and Install pygad

Clone and install the git repository:

```
$ git clone https://bitbucket.org/broett/pygad
$ cd pygad
$ sudo python setup.py install
```
I the `setuptools` module is installed, I would actually recommend `sudo python setup.py develop` (see the [wiki entry][WikiInstallation] for more).

For full functionality, you need to download the tables for [Bruzual & Charlot (2003)][BC03] SSP model and ionisation Cloudy tables (here for [Haardt & Madau, 2001][HM01]):
```
$ wget https://bitbucket.org/broett/pygad/downloads/bc03.tar.gz
$ tar -xzf bc03.tar.gz
$ wget https://bitbucket.org/broett/pygad/downloads/iontbls.tar.gz
$ tar -xzf iontbls.tar.gz
```
The [Bruzual & Charlot (2003)][BC03] tables are optional but recommended, since required for standard star plotting routines.
For absorption line generation, you need the ionisation tables.
(You might need to adjust the path to it in the `gadget.cfg`.)

If you have problems or want a more detailed explanation, see the [wiki][WikiInstallation].
We also want to point out the [FAQ section][FAQ].

### Configure

You probably need to customise [pygad]'s config files for your specific type of snapshots (even HDF5 block names can differ!).
This goes a beyond the scope of a README; be referred to the [wiki][WikiConfig].

### Use pygad

For a starter you could try something like the following in iPython:

```
#!python
import matplotlib.pyplot as plt
import pygad
import pygad.plotting   # needs to be imported explicitly
s = pygad.Snap('path/to/snap')
snap, halo = pygad.tools.prepare_zoom(s)
R200, M200 = pygad.analysis.virial_info(snap)
fig, ax, cbar = pygad.plotting.image(snap.gas, extent='5 Mpc')
ax.add_artist(plot.Circle([0,0], R200, facecolor='none', edgecolor='w'))
plt.draw()
```

There is a iPython Notebook called `QuickStart.ipynb` in the [bitbucket downloads][Downloads] that shows some more of the features.
Please, also read the [wiki][WikiHome].

iPython Notebooks can be started with

```
ipython notenook path/to/notebook.ipnb
```

---

## Support, Contact

If you have any problems, ideas, found bugs, or want to contribute in some way, please
contact me:  
[broett@mpa-garching.mpg.de](mailto:broett@mpa-garching.mpg.de)

or any of the current maintainers

[horst.foidl@outlook.com](mailto:horst.foidl@outlook.com)

[mfrigo@mpa-garching.mpg.de](mailto:mfrigo@mpa-garching.mpg.de)

[cernetic@mpa-garching.mpg.de](mailto:cernetic@mpa-garching.mpg.de)



[pygad]: https://bitbucket.org/broett/pygad
[pynbody]: https://pynbody.github.io
[HM01]: https://ui.adsabs.harvard.edu/#abs/2001cghr.confE..64H/abstract
[BC03]: http://www.bruzual.org
[Downloads]: https://bitbucket.org/broett/pygad/downloads
[WikiHome]: https://bitbucket.org/broett/pygad/wiki/Home
[WikiInstallation]: https://bitbucket.org/broett/pygad/wiki/Installation
[WikiConfig]: https://bitbucket.org/broett/pygad/wiki/Configuration
[FAQ]: https://bitbucket.org/broett/pygad/wiki/FAQ
[GSL]: http://www.gnu.org/software/gsl/
