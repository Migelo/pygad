# pygad README
[![Build Status](https://travis-ci.com/broett/pygad.svg?branch=master)](https://travis-ci.com/broett/pygad)

This module is a light-weighted (despite having ~25.000 lines of code and documentation) but comprehensive Python module that serves as a framework and basis for analysis of Gadget snapshots.

It supports all Gadget formats and is inspired and influenced by [pynbody].
[pygad] can read in Rockstar output, plot maps of any quantity, generate mock
absorption spectra, and much more.
However, its main power is its framework to conveniently analyse gadget snapshots
without the need to worry about format, read-in, indexing of blocks, units, etc.

---

## Citing pygad

If you use this code, please cite

```
RÃ¶ttgers, B., â€œLyman Î± absorption beyond the disc of simulated spiral galaxiesâ€, Monthly Notices of the Royal Astronomical Society, vol. 496, no. 1, pp. 152â€“168, 2020. doi:10.1093/mnras/staa1490. 
https://arxiv.org/abs/2005.08580
```

BibTeX for convenience:
```
@ARTICLE{2020MNRAS.496..152R,
       author = {{R{\"o}ttgers}, Bernhard and {Naab}, Thorsten and {Cernetic}, Miha and {Dav{\'e}}, Romeel and {Kauffmann}, Guinevere and {Borthakur}, Sanchayeeta and {Foidl}, Horst},
        title = "{Lyman {\ensuremath{\alpha}} absorption beyond the disc of simulated spiral galaxies}",
      journal = {\mnras},
     keywords = {line: profiles, Galaxy: formation, quasars: absorption lines, Circumstellar matter, Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = jul,
       volume = {496},
       number = {1},
        pages = {152-168},
          doi = {10.1093/mnras/staa1490},
archivePrefix = {arXiv},
       eprint = {2005.08580},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.496..152R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

---

## Getting started

### Prerequisites

Written entirely in `Python 3.x` and `C++` (the latter only for speed relevant parts), it only requires a running Python 3.x environment and the following additional quasi-standard modules:

* `numpy`
* `scipy`
* `matplotlib`
* `astropy`
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
$ sudo pip install .
```
I would actually recommend `sudo pip install -e .` (see the [wiki entry][WikiInstallation] for more). This way of installing only links the pygad folder to your site-packages. This means any changes to the code will be immediately reflected, no need for reinstallation to apply the new changes.

For full functionality, pygad will automatically download the tables for [Bruzual & Charlot (2003)][BC03] SSP model and ionisation Cloudy tables (here for [Haardt & Madau, 2001][HM01]) as well as some test snapshots and cooling function tables. The downloaded files are put in the 

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
s = pygad.Snapshot('path/to/snap')
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
contact any of the current maintainers:

[horst.foidl@outlook.com](mailto:horst.foidl@outlook.com)

[cernetic@mpa-garching.mpg.de](mailto:cernetic@mpa-garching.mpg.de)

[mfrigo@mpa-garching.mpg.de](mailto:mfrigo@mpa-garching.mpg.de)


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
