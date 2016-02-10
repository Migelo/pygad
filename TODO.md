# pygad ideas for improvement

---

### reading EAGLE snapshots

* make use of the blocks `Config`, `Constants`, `HashTable`, `Parameters`, `RuntimePars`, and `Units`
* read (Smoothed)ElementAbundance as blocks
* block 'Z' is currently understood as the 'elements' block – that's wrong!
* read snipshots...?

---

### more

- always read the default-config files and only overwrite parameters by the custom ones?!
- create images with arbitrary quantities on r-, b-, and g-channel!
- binning radially with SPH smoothing
- modules to go over:
    * analysis (some more old function are lying around...)
- utilise `__numpy_ufunc__` rather than `__array_wrap__` for add units in ufunc
  operations?
- whole Gadget IO is not well structured – do something about it
- use unit information from HDF5 files, if available
- improve writing snapshots
    * remember to write masses to header, if all of a particle type are equal
    * make it possible to write multiple files (necessary?)
