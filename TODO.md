# pygad ideas for improvement

---

### important

- fix issue #11 (utilise `__numpy_ufunc__` rather than `__array_wrap__` for
  adding units in ufunc operations)
- check if `periodic\_distance\_to` is used (rather than `dist`) where sensible!
  for `pg.plotting.image` it is not!
- binning radially with SPH smoothing (from 3D and 2D maps)
- bin onto infinitesimal thin slices (what about integral conservation?)
- cleaner way to determine block names and units
- issue #2: halo finder interface
- issue #1: Vector / streamline plots

---


### reading EAGLE snapshots

- make use of the blocks `Config`, `Constants`, `HashTable`, `Parameters`, `RuntimePars`, and `Units`
- read (Smoothed)ElementAbundance as blocks
- block 'Z' is currently understood as the 'elements' block – that's wrong!
- read snipshots...?

---

### more ideas

- create images with arbitrary quantities on r-, b-, and g-channel!
- always read the default-config files and only overwrite parameters by the custom ones?!
- go over analysis (some more old functions are lying around...)
- whole Gadget IO is not well structured – do something about it
- use unit information from HDF5 files, if available
- improve writing snapshots
    * remember to write masses to header, if all of a particle type are equal
    * make it possible to write multiple files (necessary?)
