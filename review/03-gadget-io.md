# Review: Gadget IO layer (Slice 3)

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH     | 4 |
| MEDIUM   | 5 |
| LOW      | 4 |

The Gadget binary/HDF5 I/O layer has a **critical typo** in `config.py`'s `global` declaration (`std_nameto_HDF5` instead of `std_name_to_HDF5`) that silently prevents `read_config` from properly updating the module-level dict on re-import, and a **critical `PartType` parsing bug** in `lowlevel_file.py` that only extracts the last character of the group name — breaking any HDF5 snapshot with 10+ particle types. There are also signed-vs-unsigned mismatches in `write_header`, platform-dependent `open('r')` for binary reads, a fragile SIMBA detection heuristic, and the known `Z`-block/elements misunderstanding that remains unfixed.

---

## Finding 1

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/gadget/config.py:83`

**Description:** The `global` declaration in `read_config` uses `std_nameto_HDF5` (missing underscore) instead of `std_name_to_HDF5`. The actual module-level variable is `std_name_to_HDF5` (line 59). Python's `global` statement only affects names declared in it, so the `std_name_to_HDF5.clear()` and `.update()` calls on lines 167–173 modify the module-level dict correctly *on first import* (because the `global` is not needed for mutating an existing object). However, if any code reassigns `std_name_to_HDF5` to a new dict object *outside* `read_config`, the `global` typo means `read_config` would not see the new binding. More importantly, the typo is misleading and indicates the code was never properly tested for the reassignment scenario. The current code works only because `.clear()` and `.update()` are mutations, not reassignments — but this is fragile and confusing.

**Suggested fix:** Change `std_nameto_HDF5` to `std_name_to_HDF5` in the `global` declaration on line 83.

---

## Finding 2

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/gadget/lowlevel_file.py:724`

**Description:** HDF5 particle type is parsed as `ptype = int(name[-1])` where `name` is e.g. `'PartType3'`. For particle types >= 10 (i.e., `PartType10`, `PartType11`, …), `name[-1]` extracts only the last digit: `'PartType10'[-1] == '0'`, so `ptype` becomes `0` instead of `10`. This causes particle types 10+ to be silently merged with type 0 in the block info, leading to corrupted data dimensions and incorrect ptype flags. While Gadget-2/3 standard snapshots use at most 6 types, some customized outputs or EAGLE extensions could have more.

**Suggested fix:** Use `int(name[len('PartType'):])` or `int(name.replace('PartType', ''))` to extract the full numeric suffix.

---

## Finding 3

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/gadget/lowlevel_file.py:250,254`

**Description:** `write_header` uses `struct.pack(endianness + '6i', ...)` (signed `int32`) for both `N_part` and `N_part_all`, but `read_header` (line 183,188) uses `struct.unpack(endianness + '6I', ...)` (unsigned `uint32`). If any particle count exceeds $2^{31}-1 = 2{,}147{,}483{,}647$, `struct.pack('6i', ...)` raises `struct.error` because the value exceeds the signed range. This means writing a snapshot with more than ~2.1 billion particles of any type will crash. The read path handles this correctly with unsigned format.

**Suggested fix:** Change `'6i'` to `'6I'` on lines 250 and 254 to match the read path.

---

## Finding 4

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/gadget/handler.py:135`

**Description:** `FileReader.read_block()` opens the binary file with `open(filename, 'r')` (text mode) instead of `open(filename, 'rb')` (binary mode). On Unix-like systems, this typically works because text mode and binary mode are identical. However, on Windows (and in principle any platform with newline translation), this would corrupt binary data. The constructor `__init__` correctly uses `'rb'` (line 57), making this an internal inconsistency. `np.fromfile` on line 137 technically requires binary mode.

**Suggested fix:** Change `'r'` to `'rb'` on line 135.

---

## Finding 5

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/gadget/lowlevel_file.py:170`

**Description:** SIMBA simulation detection uses a hardcoded particle count: `elif header['N_part_all'][0] == 1016261591`. This is an extremely fragile heuristic that only works for one specific SIMBA run. Any SIMBA simulation with a different number of gas particles will be misidentified (or not identified at all). Additionally, the ordering check (`flg_metals == 34` on line 173) is a Gadget-2 convention that may collide with other customized simulations.

**Suggested fix:** Remove the hardcoded particle-count check. Rely on the `flg_metals == 34` check only, or add a proper simulation-type flag to the header/config.

---

## Finding 6

**Severity:** HIGH
**Category:** correctness
**Location:** `pygad/gadget/lowlevel_file.py:508`

**Description:** Inside `_block_inferring`, line 508 reads `el_size == block.dimension * block.dtype.itemsize` — this is a comparison (`==`), not an assignment (`=`). The variable `el_size` is never assigned in this branch. The code then falls through to line 527 where `el_size` is referenced but undefined, which would raise `UnboundLocalError`. However, this code path is dead: when both `block.dimension` and `block.dtype` are known, line 546 catches the case and returns early before line 527 is reached. This is dead code with a latent bug masked by the early return.

**Suggested fix:** Either remove the dead branch (lines 507–510) or fix the assignment: `el_size = block.dimension * block.dtype.itemsize`.

---

## Finding 7

**Severity:** HIGH
**Category:** correctness
**Location:** `pygad/config/gadget.cfg:96` and `pygad/config/derived.cfg:35`

**Description:** This is the known TODO.md issue (line 24: "block 'Z' is currently understood as the 'elements' block – that's wrong!"). It remains unfixed. In `gadget.cfg`, block `Z` maps to HDF5 name `Metallicity` (a single scalar per particle). In `derived.cfg`, `elements = Z` treats this single scalar as the full multi-element abundance array. For EAGLE simulations, `Metallicity` is one dataset while element abundances are in separate `SmoothedElementAbundance*` datasets (currently silently skipped at line 732–734 of `lowlevel_file.py`). The `elements` ordering in `gadget.cfg` is also non-standard (`He, C, Mg, O, Fe, Si, H, …` instead of atomic-number order), which could cause confusion in any code that accesses element sub-arrays by index rather than name.

**Suggested fix:** As noted in TODO.md, implement proper EAGLE element reading. For the config, the element list should follow atomic-number ordering, and `Z` should not be equated to the full elements array.

---

## Finding 8

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/gadget/lowlevel_file.py:755–756`

**Description:** The block-scanning loop at line 755–756 uses `while size:` where `size` alternates between `bytes` (from `gfile.read(4)`) and `int` (from `struct.unpack`). The loop exits when `gfile.read(4)` returns `b''` (empty bytes, falsy) at EOF. However, if the file is truncated mid-block, `gfile.read(4)` could return fewer than 4 bytes (e.g., `b'\x08'`) — this is truthy, and the subsequent `struct.unpack('=i', ...)` would raise `struct.error` with a confusing message rather than a clear EOF/truncation error. Similarly, if a format-2 leading block's trailing marker is missing, the error from `assert` at line 769 would be unhelpful.

**Suggested fix:** Check that `len(size) == 4` before unpacking. Wrap the loop in a try/except with a clearer error message about file truncation.

---

## Finding 9

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/gadget/handler.py:343`

**Description:** In the `write` function, the code iterates `for name in set(blocks)-set(data.keys())` and modifies the `blocks` list inside the loop by calling `blocks.remove(name)`. Mutating a list while iterating over a *different* iterable (a `set` here) is technically safe in Python, but the pattern is fragile and confusing. More importantly, the loop prints a warning to stderr for missing blocks but does NOT raise an error — the user may not notice that expected blocks were silently skipped.

**Suggested fix:** Collect the missing block names first, print a single consolidated warning, then rebuild the `blocks` list without them. Consider raising a warning via the `warnings` module.

---

## Finding 10

**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/gadget/handler.py:168`

**Description:** There is a bare `except:` clause (line 168) that only does `raise`. This is equivalent to not having the `except` at all — it catches everything (including `SystemExit`, `KeyboardInterrupt`) and re-raises it, which is the same behavior as letting the exception propagate naturally. The `except KeyError` on line 166 is the meaningful handler; the bare `except` is dead code.

**Suggested fix:** Remove lines 168–169 entirely.

---

## Finding 11

**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/gadget/lowlevel_file.py:89–90`

**Description:** `get_format_and_endianness` returns `(3, None)` for HDF5 files, and `read_header` and `get_block_info` both check `if gformat == 3` early. However, `get_format_and_endianness` does not actually verify that the file is a valid HDF5 file when `isinstance(gfile, h5py.File)` — it just returns format 3. If someone passes a non-HDF5 file handle that happens to be an `h5py.File` object opened in write mode or an empty file, the subsequent operations would fail with confusing h5py errors rather than a clear "not a valid Gadget file" message.

**Suggested fix:** Consider adding a basic validity check (e.g., verify the `Header` group exists) when format 3 is detected.

---

## Finding 12

**Severity:** MEDIUM
**Category:** compat
**Location:** `pygad/gadget/lowlevel_file.py:151–152`

**Description:** For HDF5 files, `N_part_all` is computed as `NumPart_Total.astype(np.uint64) + (NumPart_Total_HighWord.astype(np.uint64) << 32)`. The `astype(np.uint64)` is correct and handles NumPy 2.x fine. However, the `list()` wrapper on line 151 converts this to a Python list of numpy uint64 scalars. When this list is later used in `struct.pack('6i', ...)` (Finding 3) or compared with Python ints, the numpy scalar types could cause subtle issues in edge cases. This is more of a type-safety concern than a current bug.

**Suggested fix:** Convert to native Python `int` values: `[int(x) for x in ...]`.

---

## Finding 13

**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/gadget/handler.py:355`

**Description:** In the HDF5 write path (lines 353–355), `units` is set inside a loop over particle types: `for pt in range(6): if d[pt] is not None: units = d[pt].units`. If all `d[pt]` are `None` (which would be unusual but possible if the block was somehow added to `blocks` but has no data for any ptype), `units` would be undefined when referenced at line 364, causing `UnboundLocalError`.

**Suggested fix:** Initialize `units` before the loop, e.g., `units = Unit(1)`.

---

## Finding 14

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/gadget/config.py:118–120`

**Description:** The block_order and elements lists are cleared and rebuilt using `while list: list.pop()` followed by `list += [...]`. This is functionally correct but unnecessarily convoluted. Since `read_config` already calls `.clear()` on the dicts, it should also just reassign the lists.

**Suggested fix:** Replace the `while ... pop()` idiom with simple reassignment: `block_order = [...]`.

---

## Finding 15

**Severity:** LOW
**Category:** qol
**Location:** `pygad/gadget/lowlevel_file.py:810`

**Description:** `_write_format2_leading_block` uses `struct.pack(endianness + ' i 4s i i', ...)` with a leading space in the format string. While `struct.pack` ignores whitespace in format strings (it's valid), this is unusual and could confuse readers.

**Suggested fix:** Remove the extraneous space: `endianness + 'i4sii'`.

---

## Finding 16

**Severity:** LOW
**Category:** correctness
**Location:** `pygad/gadget/lowlevel_file.py:169`

**Description:** `header['flg_arepo']` is set based on `gfile['Config'].attrs.__contains__("VORONOI")`. The `__contains__` dunder method usage is unnecessary and non-idiomatic. More importantly, this relies on the `Config` group existing in the HDF5 file. For non-AREPO simulations that happen to have a `Config` group without `VORONOI`, `flg_arepo` is set to `False` — which is correct but the detection relies on the `Config` group being a reliable AREPO indicator, which is not guaranteed.

**Suggested fix:** Use `"VORONOI" in gfile['Config'].attrs` instead of `.__contains__`.

---

## Finding 17

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/gadget/handler.py:356`

**Description:** In the HDF5 write path, `hdf5name = snap._root._load_name.get(name)` retrieves the padded standard name, then looks it up in `std_name_to_HDF5`. But `std_name_to_HDF5` is built from `gadget.cfg`'s `[hdf5 names]` section. If a block name is not in `_load_name` (e.g., a custom block), the fallback `hdf5name = name` uses the raw attribute name (e.g., `'myblock'`) which will not match any dataset in the HDF5 file. This should be documented or raise a clearer error.

**Suggested fix:** Add a check: if `hdf5name is None` after the `_load_name` lookup and the block is not in `std_name_to_HDF5`, issue a warning that the HDF5 dataset name is being guessed.
