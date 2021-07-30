'''
Some general high-level functions.

Example:
'''
__all__ = ['read_info_file', 'prepare_zoom', 'fill_star_from_info',
           'read_traced_gas', 'fill_gas_from_traced',
           'fill_derived_gas_trace_qty', 'read_pyigm_COS_halos_EWs']

from .snapshot import *
from .units import *
from .utils import *
from .analysis import *
from .transformation import *
from . import environment
import re
import sys
import os
import numpy as np


def read_info_file(filename):
    '''
    Read in the contents of an info file as produced by gtrace into a dictionary.

    It is assumed, that there is exactly one colon per line, which separates the
    name (the part before the colon) and the value (the part after the colon). For
    the value part the following is assumed: if there are no brackets ('[') it is
    a float; if there is one pair of brackets, it is a float with units, which are
    given within the brackets; otherwise the value is simply stored as a string.

    Args:
        filename (str):     The path to the info file.

    Returns:
        info (dict):        A dictionary containing all the entries from the file.
    '''
    info = {}
    with open(os.path.expanduser(filename), 'r') as finfo:
        for line in finfo:
            try:
                cols = line.split(':', 1)
                if len(cols) > 1:           # otherwise no value given, avoid ValueError
                    name = cols[0].strip()
                    value = cols[1].strip()
                    blocks = re.findall('\[(.*?)\]', value)
                    if len(blocks) == 0:
                        value = float(value) if value != 'None' else None
                    elif len(blocks) == 1:
                        value = UnitArr(float(value.rsplit('[')[0].strip()),
                                        units=blocks[0])
                    elif len(blocks) == 2:
                        value = UnitArr([float(s.strip(', ')) for s in blocks[0].split()],
                                        units=None if blocks[1] == 'None' else blocks[1])
                    if name in info:
                        print('WARNING: "%s" occures ' % name + \
                              'multiple times in info file ' + \
                              '"%s"! First one used.' % filename, file=sys.stderr)
                    info[name] = value
            except ValueError as e:
                continue    # ignore error continue loading

    return info


def prepare_zoom(s, mode='auto', info='deduce', shrink_on='stars',
                 linking_length=None, linking_vel='200 km/s', ret_FoF=False,
                 sph_overlap_mask=False, gal_R200=0.10, star_form='deduce',
                 gas_trace='deduce', to_physical=True, load_double_prec=False,
                 fill_undefined_nan=True, gas_traced_blocks='all',
                 gas_traced_dervied_blocks=None, center_on_BH=False, **kwargs):
    '''
    A convenience function to load a snapshot from a zoomed-in simulation that is
    not yet centered or oriented.

    Args:
        s (str, Snap):      The snapshot of a zoomed-in simulation to prepare.
                            Either as an already loaded snapshot or a path to the
                            snapshot.
        mode (str):         The mode in which to prepare the snapshot. You can
                            choose from:
                                * 'auto':   try 'info', if it does not work
                                            fallback to 'FoF'
                                * 'info':   read the info file (from `info`) and
                                            take the center and the reduced
                                            inertia tensor (for orientation) with
                                            fallback to angular momentum from this
                                            file
                                * 'ssc':    center the snapshot using a
                                            shrinking sphere method (shrinking on
                                            the sub-snapshot specified by
                                            `shrink_on`, see below)
                                            ATTENTION: can easily center on a
                                            non-resolved galaxy!
                                * 'FoF':    Do a FoF search on all particles and
                                            generate a halo catalogue for all
                                            those halos that have at most 1% mass
                                            from "lowres" particles. Then find the
                                            most massive FoF group of particles
                                            defined by `shrink_on` therein and
                                            finally perform a shrinking sphere on
                                            them to find the center. If
                                            `shrink_on` is 'all' or 'highres',
                                            skip the second FoF search and do the
                                            "ssc" on the most massive FoF group of
                                            the first step. The linking length for
                                            (both) the FoF finder is given by
                                            `linking_length`.
        info (str, dict):   Path to info file or the dictionary as returned from
                            `read_info_file`.
                            However, if set to 'deduce', it is tried to deduce the
                            path to the info file from the snapshot filename: it
                            is assumed to be in a subfolder 'trace' that is in the
                            same directory as the snapshot and named
                            `info_%03d.txt`, where the `%03d` is filled with the
                            snapshot number. The latter is taken as the last three
                            characters of the first dot / the end of the filename.
        shrink_on (str, list):
                            Define the part of the snapshot to perform the
                            shrinking sphere on, if there is no info file found or
                            info==None. It can be 'all' (use the entire
                            (sub-)snapshot `s`), a family name, or a list of
                            particle types (e.g. [0,1,4]).
        linking_length (UnitScalar):
                            The linking length used for the FoF finder in mode
                            "FoF" (if None, it defaults to
                            ( 1500.*rho_crit / median(mass) )^(-1/3) as defined in
                            `generate_FoF_catalogue`).
        linking_vel (UnitScalar):
                            The linking velocity used for the FoF finder in mode
                            "FoF". Only used for defining the galaxy, not for the
                            halo.
        ret_FoF (bool):     Also return the FoF group catalogue created (if in
                            mode "FoF").
        sph_overlap_mask (bool):
                            Whether to mask all particles that overlap into the
                            halo region (that is also include SPH particles that
                            are outside the virial radius, but their smoothing
                            length reaches into it).
        gal_R200 (float):   The radius to define the galaxy. Everything within
                            <gal_R200>*R200 will be defined as the galaxy.
        star_form (str):    Path to the star formation file as written by the
                            program `gtrace`.
                            If set to 'deduce', its path is tried to build from
                            the snapshot filename by taking its directory and
                            adding '/trace/star_form.ascii'.
        gas_trace (str):    Path to the tracing file as written by the program
                            `gtracegas`.
                            If set to 'deduce', its path is tried to build from
                            the snapshot filename by taking its directory and
                            looking for files '/../*gastrace*'.
        to_physical (bool): Whether to convert the snapshot to physical units.
        load_double_prec (bool):
                            Force to load all blocks in double precision.
                            Equivalent with setting the snapshots attribute.
        fill_undefined_nan (bool):
                            Fill star formation info for IDs not listed in the
                            `star_form` file. For more info see
                            `fill_star_from_info`.
        gas_traced_blocks (iterable, str):
                            The blocks to add with `fill_gas_from_traced`.
        gas_traced_dervied_blocks (bool):
                            Passed to `fill_gas_from_traced` as `add_derived`.
                            Defaults to `gas_traced_blocks=='all'` if None.
        center_on_BH (bool):
                    If True, positions and velocities are centered on
                    the central supermassive black hole (if present).
                    If there are multiple supermassive black holes
                    within the central kpc of the galaxy, their center
                    of mass is used as the center.
        kwargs:             Passed to `generate_FoF_catalogue` in `mode='FoF'`.

    Returns:
        s (Snap):           The prepared snapshot.
        halo (SubSnap):     The cut halo of the found structure.
        gal (SubSnap):      The central galaxy of the halo as defined by
                            `gal_R200`.
    '''

    def get_shrink_on_sub(snap, shrink_on):
        if isinstance(shrink_on, str):
            shrink_on = str(shrink_on)
            if shrink_on == 'all':
                return snap
            else:
                return getattr(s, shrink_on)
        elif isinstance(shrink_on, list):
            return s[shrink_on]
        else:
            raise ValueError('`shrink_on` must be a family name or a list ' + \
                             'of particle types, but was: %s' % (shrink_on,))

    def FoF_exclude(h, s, threshold=1e-2):
        M_lowres = h.lowres_mass
        M = h.mass
        return M_lowres / M > threshold

    if 'gastrace' in kwargs:
        raise ValueError("You passed 'gastrace'. Did you mean 'gas_trace'?")
    if 'starform' in kwargs:
        raise ValueError("You passed 'starform'. Did you mean 'star_form'?")

    if isinstance(s, str):
        s = Snapshot(s, load_double_prec=load_double_prec)
    gal_R200 = float(gal_R200)
    if environment.verbose >= environment.VERBOSE_TACITURN:
        print('prepare zoomed-in', s)

    # read info file (if required)
    if mode in ['auto', 'info']:
        if info == 'deduce':
            try:
                snap = int(os.path.basename(s.filename).split('.')[0][-3:])
                info = os.path.dirname(s.filename) + '/trace/info_%03d.txt' % snap
            except:
                print('WARNING: could not deduce the path to ' + \
                      'the trace file!', file=sys.stderr)
                info = None
        if isinstance(info, str):
            info = os.path.expanduser(info)
            if not os.path.exists(info):
                print('WARNING: There is no info file named ' + \
                      '"%s"' % info, file=sys.stderr)
                info = None
            else:
                if environment.verbose >= environment.VERBOSE_TACITURN:
                    print('read info file from:', info)
                info = read_info_file(info)
        if info is None:
            if mode == 'auto':
                mode = 'FoF'
            else:
                raise IOError('Could not read/find the info file!')
        else:
            if mode == 'auto':
                mode = 'info'

    if to_physical:
        s.to_physical_units()

    # find center
    if mode == 'info':
        center = info['center']
    elif mode in ['ssc', 'FoF']:
        if mode == 'FoF':
            halos = generate_FoF_catalogue(
                s,
                l=linking_length,
                exclude=FoF_exclude,
                calc=['mass', 'lowres_mass'],
                max_halos=10,
                progressbar=False,
                **kwargs
            )
            if shrink_on not in ['all', 'highres']:
                galaxies = generate_FoF_catalogue(
                    get_shrink_on_sub(s, shrink_on),
                    l=linking_length,
                    dvmax=linking_vel,
                    calc=['mass', 'com'],
                    max_halos=10,
                    progressbar=False,
                    **kwargs
                )
                # The most massive galaxy does not have to be in a halo with litle
                # low resolution elements! Find the most massive galaxy living in a
                # "resolved" halo:
                galaxy = None
                for gal in galaxies:
                    # since the same linking lengths were used for the halos and
                    # the galaxies (rethink that!), a galaxy in a halo is entirely
                    # in that halo or not at all
                    gal_ID_set = set(gal.IDs)
                    for h in halos:
                        if len(gal_ID_set - set(h.IDs)) == 0:
                            galaxy = gal
                            break
                    del h, gal_ID_set
                    if galaxy is not None:
                        break
                del galaxies
                if galaxy is None:
                    shrink_on = None
                else:
                    shrink_on = s[galaxy]
            else:
                shrink_on = s[halos[0]]
        elif mode == 'ssc':
            shrink_on = get_shrink_on_sub(s, shrink_on)
        if shrink_on is not None and len(shrink_on) > 0:
            com = center_of_mass(s)
            R = np.max(periodic_distance_to(s['pos'], com, s.boxsize))
            center = shrinking_sphere(shrink_on, com, R)
        else:
            center = None
    else:
        raise ValueError('Unkown mode "%s"!' % mode)

    # center in space
    if center is None:
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print('no center found -- do not center')
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('center at:', center)
        Translation(-center).apply(s)
        # center the velocities
        vel_center = mass_weighted_mean(s[s['r'] < '1 kpc'], 'vel')
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('center velocities at:', vel_center)
        s['vel'] -= vel_center

    # center galaxy on central supermassive black hole(s)
    if center_on_BH:
        bh_search_rad_kpc = 1.
        search_ball=s[BallMask(str(bh_search_rad_kpc)+' kpc')]
        if search_ball.bh["mass"].size==0: #No BH
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print("No black holes found, center stays the same.")
        else:
            if search_ball.bh["mass"].size > 1: #>1 BHs
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print("WARNING: Multiple black holes within the central kpc, centering on their center of mass")
                #selecting the most massive black hole
                bhpos = np.average(search_ball.bh["pos"], axis=0,
                                   weights=search_ball.bh["mass"] )
                bhvel = np.average(search_ball.bh["vel"], axis=0,
                                   weights=search_ball.bh["mass"] )
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print("Center of mass: " + str(bhpos))
                    print("Velocity of center of mass: " + str(bhvel))
            else:  #1 BH
                bhpos = search_ball.bh["pos"][0]
                bhvel = search_ball.bh["vel"][0]
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print("Central black hole position: " + str(bhpos))
                    print("Central black hole velocity: " + str(bhvel))
            s["pos"] = s["pos"] - bhpos
            s["vel"] = s["vel"] - bhvel


    # cut the halo (<R200)
    if mode == 'info':
        R200 = info['R200']
        M200 = info['M200']
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('derive virial information')
        R200, M200 = virial_info(s)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('R200:', R200)
        print('M200:', M200)
    halo = s[BallMask(R200, sph_overlap=sph_overlap_mask)]

    # orientate at the reduced inertia tensor of the baryons wihtin 10 kpc
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('orientate', end=' ')
    if mode == 'info':
        if 'I_red(gal)' in info:
            redI = info['I_red(gal)']
            if redI is not None:
                redI = redI.reshape((3, 3))
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('at the galactic red. inertia tensor from info file')
            if environment.verbose >= environment.VERBOSE_TALKY:
                print(redI)
            mode, qty = 'red I', redI
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('at angular momentum of the galaxtic baryons from info file:')
            mode, qty = 'vec', info['L_baryons']
        orientate_at(s, mode, qty=qty, total=True)
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('at red. inertia tensor of the baryons within %.3f*R200' % gal_R200)
        orientate_at(shrink_on[BallMask(gal_R200*R200, sph_overlap=False)],
                     'red I',
                     total=True
                     )

    # cut the inner part as the galaxy
    gal = s[BallMask(gal_R200 * R200, sph_overlap=sph_overlap_mask)]
    Ms = gal.stars['mass'].sum()
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('M*:  ', Ms)

    if len(gal) == 0:
        gal = None
    if len(halo) == 0:
        halo = None

    if star_form == 'deduce':
        try:
            star_form = os.path.dirname(s.filename) + '/trace/star_form.ascii'
        except:
            print('WARNING: could not deduce the path to the ' + \
                  'star formation file!', file=sys.stderr)
            star_form = None
    if isinstance(star_form, str):
        star_form = os.path.expanduser(star_form)
        if not os.path.exists(star_form):
            print('WARNING: There is no star formation file ' + \
                  'named "%s"' % star_form, file=sys.stderr)
            star_form = None
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('read star formation file from:', star_form)
            fill_star_from_info(s, star_form,
                                fill_undefined_nan=fill_undefined_nan)

    if gas_trace == 'deduce':
        try:
            directory = os.path.dirname(s.filename) + '/../'
            candidates = []
            for fname in os.listdir(directory):
                if fname.startswith('gastrace'):
                    candidates.append(fname)
            if len(candidates) == 1:
                gas_trace = directory + candidates[0]
            else:
                raise RuntimeError('too many candidates!')
        except:
            print('WARNING: could not deduce the path to the ' + \
                  'gas tracing file!', file=sys.stderr)
            gas_trace = None
    if isinstance(gas_trace, str):
        gas_trace = os.path.expanduser(gas_trace)
        if not os.path.exists(gas_trace):
            print('WARNING: There is no gas trace file named ' + \
                  '"%s"' % gas_trace, file=sys.stderr)
            gas_trace = None
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('read gas trace file from:', gas_trace)
            if gas_traced_dervied_blocks is None:
                gas_traced_dervied_blocks = (gas_traced_blocks == 'all')
            fill_gas_from_traced(s, gas_trace,
                                 add_blocks=gas_traced_blocks,
                                 add_derived=gas_traced_dervied_blocks)

    if mode == 'FoF' and ret_FoF:
        return s, halo, gal, halos
    else:
        return s, halo, gal


def fill_star_from_info(snap, fname, fill_undefined_nan=True, dtypes=None,
                        units=None):
    '''
    Read the formation radius rform and rform/R200(aform) from the star_form.ascii
    file and create the new blocks "rform" and "rR200form".

    Note:
        The set of the star particle IDs in the star formation information file
        must exactly be the set of star particle IDs in the (root) snapshot.

    Args:
        snap (Snap):    The snapshot to fill with the data (has to be the one at
                        z=0 of the simulation used to create the star_form.ascii).
        fname (str):    The path to the star_form.ascii file.
        fill_undefined_nan (bool):
                        Fill data with NaN for stars that are not listed in the
                        formation file.
        dtypes (np.dtypes):
                        The named(!) fields in the formation file. One of the has
                        to be a (integer type) field called "ID" for matching the
                        data to the snapshot.
        units (dict):   Specifying the units of the fields. Not every field has to
                        be sopecified here.
                        Defaults to: `{'aform':'a_form', 'rform':'kpc'}`.

    Raises:
        RuntimeError:   If the IDs are not unique or they do not match (except the
                        cases where `fill_undefined_nan` applies).
    '''
    stars = snap.root.stars

    if environment.verbose >= environment.VERBOSE_TACITURN:
        print('reading the star formation information from %s...' % fname)
    # prepare the type of data
    if dtypes is None:
        dtypes = [('ID', np.uint64), ('aform', float), ('rform', float),
                  ('rR200form', float), ('Zform', float)]
    dtypes = np.dtype(dtypes)
    if 'ID' not in dtypes.fields:
        raise ValueError('The `dtypes` need to have a field "ID"!')
    if units is None:
        units = {'aform': 'a_form', 'rform': 'kpc'}
    # load the data
    SFI = np.loadtxt(fname, skiprows=1, dtype=dtypes)

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('testing if the IDs match the (root) snapshot...')
    SFI_IDs = SFI['ID']
    # test uniqueness
    if not stars.IDs_unique():
        raise RuntimeError('Stellar IDs in the snapshot are not unique!')
    if len(np.unique(SFI_IDs)) != len(SFI_IDs):
        raise RuntimeError('IDs in the star formation file are not unique!')
    # there might be too many or not enough IDs in the file
    if len(np.setdiff1d(stars['ID'], SFI_IDs, assume_unique=True)):
        missing = np.setdiff1d(stars['ID'], SFI_IDs, assume_unique=True)
        if fill_undefined_nan:
            print('WARNING: There are %d stellar IDs missing ' % len(missing) + \
                  'in the formation file! Fill them with NaN\'s.')
            add = np.array([(ID,) + (np.NaN,) * (len(dtypes) - 1) for ID in missing],
                           dtype=dtypes)
            SFI = np.concatenate((SFI, add))
            del add
            SFI_IDs = SFI['ID']
        else:
            raise RuntimeError('Some stars do not have a match in the ' + \
                               'formation file "%s" (missing: %d)!' % (fname,
                                                                       len(np.setdiff1d(stars['ID'], SFI_IDs,
                                                                                        assume_unique=True)))
                               )
    if len(np.setdiff1d(SFI_IDs, stars['ID'], assume_unique=True)):
        raise RuntimeError('Some formation file IDs do not have a match in ' + \
                           'the snapshot (missing: %d)!' % (
                               len(np.setdiff1d(SFI_IDs, stars['ID'],
                                                assume_unique=True)))
                           )

    # adding the data as blocks
    sfiididx = np.argsort(SFI_IDs)
    sididx = np.argsort(stars['ID'])
    for name in dtypes.names:
        if name == 'ID':
            continue
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('adding the new block "%s" (units:%s, type:%s)...' % (
                name, units.get(name, None), dtypes[name]))
        stars[name] = UnitArr(np.empty(len(stars)),
                              dtype=dtypes[name],
                              units=units.get(name, None))
        stars[name][sididx] = SFI[name][sfiididx]


def read_traced_gas(filename, types=None):
    '''
    Read the gas tracing statistics from a gtracegas output.

    The data also gets tagged by type:
        1:  gas in region
        2:  gas out of region
        3:  stars that formed in region
        4:  stars that formed outside region
    and the number of full cycles (leaving region + re-entering).

    TODO:
        * fill the stars that have been traced gas!

    The following is tracked at the different events:
        (re-)entering the region:       [a, mass, metals, j_z, T]
        the leave the region:           [a, mass, metals, j_z, T, vel]
        turning into a star:            [a, mass, metals, j_z]
        being out of the region, the following is updated:
                                        [(a, r_max), (a, z_max)]
    where a is the scale factor of when the particle reached the maximum radius or
    height, respectively.

    Args:
        filename (str):     The filename to read from.
        types (iterable):   The types of particles (see above!) to filter for.
                            Default: all.

    Returns:
        tr (dict):          A dictionary with the trace data for the individual
                            particles (of types ask for) indexed by ID. The actual
                            data is:
                                [ [type, n],
                                  np.ndarray of first enter event,
                                  np.ndarray leaving event,          \
                                  np.ndarray of traced being out,     >  n times
                                  np.ndarray of re-entering,         /
                                  events of leaving + out,          \  depending
                                  forming a star, or                 > on type or
                                  leaving + out + forming a stars,  / just nothing
                                ]
                            where n is the number of full recylces.
    '''
    filename = os.path.expanduser(filename)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('read gas trace file "%s"...' % filename)
        sys.stdout.flush()
    import pickle as pickle
    with open(filename, 'rb') as f:
        tr = pickle.load(f)

    # sort type
    if types is None or types == 'all':
        types = {1, 2, 3, 4}
    else:
        if isinstance(types, int):
            types = [types]
        types = set(types)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print('restructure by type...')
        sys.stdout.flush()
    # structure the data into sub-lists
    # (re-)enter:   5 elements
    # leave:        6 elements
    # star form:    4 elements
    # being out:    4 elements
    # -> full recycle:  6+4+5 = 15
    # -> in region:     5 + n*15                =  5 + n*15
    #    out:           5 + n*15 + 6 + 4        = 15 + n*15
    #    SF in region:  5 + n*15 + 4            =  9 + n*15
    #    SF outside:    5 + n*15 + 6 + 4 + 4    = 19 + n*15
    t_to_type = {5: 1, 15: 2, 9: 3, 19: 4}
    for ID in list(tr.keys()):
        e = tr[ID]
        t = len(e) % 15
        if t in [0, 4]: t += 15
        tt = t_to_type[t]
        if tt not in types:
            print('ERROR: could not process particle ID %d' % ID, file=sys.stderr)
            print('       skip: %s' % e, file=sys.stderr)
            del tr[ID]
            continue
        sub = e[5: {5: None, 15: -10, 9: -4, 19: -14}[t]]
        # assert len(sub) % 15 == 0
        new = [[tt, len(sub) / 15]]
        new += [e[:5]]
        for re in np.array(sub).reshape(((len(e) - t) / 15, 15)):
            new += [re[:6]]
            new += [re[6:10]]
            new += [re[10:]]
        if t == 5:  # gas in region (with possible recycles)
            pass
        elif t == 9:  # turned in the region into a star
            new += [e[-4:]]
        elif t == 15:  # left region, but is still a gas particle
            new += [e[-10:-4], e[-4:]]
        elif t == 19:  # left region and turned into a star
            new += [e[-14:-8], e[-8:-4], e[-4]]
        else:
            raise RuntimeError('Structure in "%s" ' % filename + \
                               'is not as expected!')
        tr[ID] = new

    return tr


def fill_gas_from_traced(snap, data, add_blocks='all', add_derived=True,
                         units=None, invalid=0.0):
    '''
    Fill some information from the gas trace file into the snapshot as blocks.

    This function is using data from a gas trace file (as produced by `gtracegas`)
    to create new gas blocks with particle properties at in- and outflow times as
    well as additional information of the time being ejected.

    Among the new blocks are:
      "trace_type", "num_recycled",
      "infall_a", "infall_time",
      "mass_at_infall", "metals_at_infall", "jz_at_infall", "T_at_infall",
      "ejection_a", "ejection_time",
      "mass_at_ejection", "metals_at_ejection", "jz_at_ejection", "T_at_ejection",
      "cycle_r_max_at", "cycle_r_max", "cycle_z_max_at", "cycle_z_max",
    For the addtional derived blocks see `fill_derived_gas_trace_qty`.

    Note:
        The set of the star particle IDs in the star formation information file
        must exactly be the set of star particle IDs in the (root) snapshot.

    Args:
        snap (Snap):        The snapshot to fill with the data (has to be the one
                            at z=0 of the simulation used to create the trace
                            file).
        data (str, dict):   The path to the gas trace file or the alread read-in
                            data.
        add_blocks (iterable, str):
                            A list of the block names to actually add to the gas
                            sub-snapshot, or just 'all', which then adds all of
                            them. If `add_derived` it True, it will be set to 'all'
                            anyway, since they might be needed afterwards.
        add_derived (bool): Whether to also add blocks that can be derived from
                            the trace data stored in blocks. The function
                            `fill_derived_gas_trace_qty` is used for that.
        units (dict):       The units to use for masses, lengths, etc..
                            Defaults to: {'TIME':'a_form', 'MASS':'Msol',
                                          'TEMP':'K', 'ANGMOM':None, 'POS':'kpc'}
        invalid (float/NaN/inf):
                            The value to fill invalid entries with. Such entries
                            are properties of cycles that did not happen for a
                            given particle.
    '''
    if add_derived:
        add_blocks = 'all'
    if add_blocks == 'all':
        add_blocks = ["trace_type", "num_recycled",
                      "infall_a", "infall_time",
                      "mass_at_infall", "metals_at_infall", "jz_at_infall", "T_at_infall",
                      "ejection_a", "ejection_time",
                      "mass_at_ejection", "metals_at_ejection", "jz_at_ejection", "T_at_ejection",
                      "cycle_r_max_at", "cycle_r_max", "cycle_z_max_at", "cycle_z_max",
                      ]
    if isinstance(add_blocks, str):
        add_blocks = (add_blocks,)
    if units is None:
        # TODO: angmom units
        units = dict(TIME='a_form', MASS='Msol', TEMP='K',
                     ANGMOM=None, POS='kpc')
    gas = snap.root.gas
    environment.gc_full_collect()

    if isinstance(data, str):
        filename = data
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print('reading the gas trace information from %s...' % filename)
        data = read_traced_gas(data)
    else:
        filename = '<given data>'

    # filter to gas only
    gas_type = {1, 2}
    data = dict([i for i in iter(data.items()) if i[1][0][0] in gas_type])

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('test IDs and find matching IDs...')
    if len(set(data.keys()) - set(gas['ID'])) > 0:
        raise RuntimeError('Traced gas IDs in "%s" have ' % filename +
                           '%s ' % nice_big_num_str(len(
            set(data.keys()) - set(gas['ID']))) +
                           'elements that are not in the snapshot!')
    tracedIDs = (set(data.keys()) & set(gas['ID']))

    trmask = np.array([(ID in tracedIDs) for ID in gas['ID']], dtype=bool)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print('  found %s (of %s)' % (nice_big_num_str(len(tracedIDs)),
                                      nice_big_num_str(len(data))), end=' ')
        print('traced IDs that are in the snapshot')

    def add_block(name, block):
        if name in add_blocks:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('  "%s"' % name)
            gas[name] = block

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('adding the blocks:')

    trididx = np.argsort(list(data.keys()))
    gididx = np.argsort(gas['ID'])
    gididx_traced = gididx[trmask[gididx]]

    # type = not traced (0), in region (1), out of region (2)
    # number of full cycles (out and in)

    trace_type = UnitArr(np.zeros(len(gas), dtype=int))
    trtype = np.array([i[0][0] for i in data.values()])
    trace_type[gididx_traced] = trtype[trididx]
    add_block('trace_type', trace_type)
    del trace_type

    n_cyc = np.array([i[0][1] for i in data.values()])
    num_recycled = UnitArr(np.empty(len(gas), dtype=n_cyc.dtype))
    num_recycled[~trmask] = -1
    num_recycled[gididx_traced] = n_cyc[trididx]
    add_block('num_recycled', num_recycled)
    set_N_cycles = set(n_cyc)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print('  +++ number of recycles that occured:', set_N_cycles, '+++')

    # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
    # events of entering the region. N is the number of the gas particles. Each
    # particle has entries for each of their enter events, those events that do
    # not exists, get filled with `invalid`.

    max_N_cycle = max(set_N_cycles)
    infall_t = np.array([[e[0] for e in i[1:1 + 3 * n + 1:3]] + [invalid] * (max_N_cycle - n)
                         for n, i in zip(n_cyc, list(data.values()))])
    infall_a = UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                dtype=infall_t.dtype), units=units['TIME'])
    infall_a[~trmask] = invalid
    infall_a[gididx_traced] = infall_t[trididx]
    add_block('infall_a', infall_a)
    del infall_t
    environment.gc_full_collect()

    from .snapshot import age_from_form
    # only convert reasonable values & ensure not to overwrite blocks
    mask = (infall_a != invalid) & np.isfinite(infall_a)
    infall_time = infall_a.copy()
    new = gas.cosmic_time() - age_from_form(infall_time[mask], subs=gas)
    infall_time.units = new.units
    infall_time[mask] = new
    del new
    add_block('infall_time', infall_time)
    del infall_time
    environment.gc_full_collect()

    for name, idx, unit in [('mass_at_infall', 1, units['MASS']),
                            ('metals_at_infall', 2, units['MASS']),
                            ('jz_at_infall', 3, units['ANGMOM']),
                            ('T_at_infall', 4, units['TEMP'])]:
        if name not in add_blocks:
            continue
        infall_Q = np.array([[e[idx] for e in i[1:1 + 3 * n + 1:3]] +
                             [invalid] * (max_N_cycle - n)
                             for n, i in zip(n_cyc, list(data.values()))])
        block = UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                 dtype=infall_Q.dtype),
                        units=unit)
        block[~trmask] = invalid
        block[gididx_traced] = infall_Q[trididx]
        add_block(name, block)
        del infall_Q, block
        environment.gc_full_collect()

    # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
    # events of ejection / leaving the region. N is the number of the gas
    # particles. Each particle has entries for each of their ejection events,
    # those events that do not exists, get filled with `invalid`.
    max_N_cycle = max(set_N_cycles)
    eject_t = np.array([[e[0] for e in i[2:2 + 3 * n:3]] +
                        [i[2 + 3 * n][0] if t == 2 else invalid] +
                        [invalid] * (max_N_cycle - n)
                        for n, t, i in zip(n_cyc, trtype, list(data.values()))])
    ejection_a = UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                  dtype=eject_t.dtype),
                         units=units['TIME'])
    ejection_a[~trmask] = invalid
    ejection_a[gididx_traced] = eject_t[trididx]
    del eject_t
    add_block('ejection_a', ejection_a)
    environment.gc_full_collect()

    from .snapshot import age_from_form
    # only convert reasonable values & ensure not to overwrite blocks
    mask = (ejection_a != invalid) & np.isfinite(ejection_a)
    ejection_time = ejection_a.copy()
    new = gas.cosmic_time() - age_from_form(ejection_time[mask], subs=gas)
    ejection_time.units = new.units
    ejection_time[mask] = new
    del new
    add_block('ejection_time', ejection_time)
    environment.gc_full_collect()

    for name, idx, unit in [('mass_at_ejection', 1, units['MASS']),
                            ('metals_at_ejection', 2, units['MASS']),
                            ('jz_at_ejection', 3, units['ANGMOM']),
                            ('T_at_ejection', 4, units['TEMP'])]:
        if name not in add_blocks:
            continue
        eject_Q = np.array([[e[idx] for e in i[2:2 + 3 * n:3]] +
                            [i[2 + 3 * n][idx] if t == 2 else invalid] +
                            [invalid] * (max_N_cycle - n)
                            for n, t, i in zip(n_cyc, trtype, list(data.values()))])
        block = UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                 dtype=eject_Q.dtype),
                        units=unit)
        block[~trmask] = invalid
        block[gididx_traced] = eject_Q[trididx]
        add_block(name, block)
        del eject_Q, block
        environment.gc_full_collect()

    # for each cycle ther is a maximum travel distance, plus one more for those
    # particles that are outside the region: store them
    for name, idx, unit in [('cycle_r_max_at', 0, units['TIME']),
                            ('cycle_r_max', 1, units['POS']),
                            ('cycle_z_max_at', 2, units['TIME']),
                            ('cycle_z_max', 3, units['POS'])]:
        if name not in add_blocks:
            continue
        pos = np.array([[e[idx] for e in i[3:3 + 3 * n:3]] +
                        [i[3 + 3 * n][idx] if t == 2 else invalid] +
                        [invalid] * (max_N_cycle - n)
                        for n, t, i in zip(n_cyc, trtype, list(data.values()))])
        block = UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                 dtype=pos.dtype),
                        units=unit)
        block[~trmask] = invalid
        block[gididx_traced] = pos[trididx]
        add_block(name, block)
        del pos, block
        environment.gc_full_collect()

    if add_derived:
        fill_derived_gas_trace_qty(snap, units=units, invalid=invalid)


def fill_derived_gas_trace_qty(snap, units=None, invalid=0.0):
    """
    Derive blocks from the gas trace blocks and add them.

    Among the new blocks are:
      "out_time",
      "last_infall_a", "last_infall_time", "mass_at_last_infall",
        "metals_at_last_infall", "jz_at_last_infall", "T_at_last_infall",
      "last_ejection_a", "last_ejection_time", "mass_at_last_ejection",
        "metals_at_last_ejection", "jz_at_last_ejection", "T_at_last_ejection"'

    Note:
        The arguments `units` and `invalid` shall be the same as for
        `fill_gas_from_traced`.
        Also, some blocks added by `fill_gas_from_traced` are needed. Make sure
        they are avaiable!

    Args:
        snap (Snap):        The snapshot to fill with the data (has to be the one
                            at z=0 of the simulation used to create the trace
                            file).
        units (dict):       The units to use for masses, lengths, etc..
        invalid (float/NaN/inf):
                            The value to fill invalid entries with. Such entries
                            are properties of cycles that did not happen for a
                            given particle.
    """
    if units is None:
        # TODO: angmom units
        units = dict(TIME='a_form', MASS='Msol', TEMP='K',
                     ANGMOM=None, POS='kpc')
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('adding blocks that can be derived from the gas trace blocks:')

    gas = snap.gas

    trmask = (gas['num_recycled'] != -1)
    set_N_cycles = set(gas['num_recycled'])
    max_N_cycle = max(set_N_cycles)

    # each (full) cycle takes some given time
    if max_N_cycle > 0:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('  "out_time",')
        ejected = gas['ejection_time'][:, :-1]
        infall = gas['infall_time'][:, 1:]
        gas['out_time'] = infall - ejected
        mask = (ejected == invalid) | ~np.isfinite(ejected)
        del ejected
        environment.gc_full_collect()
        gas['out_time'][mask] = invalid
        mask = (infall == invalid) | ~np.isfinite(infall)
        del infall
        environment.gc_full_collect()
        gas['out_time'][mask] = invalid
    environment.gc_full_collect()

    # The events of the last infall and the last ejection are a bit messy to
    # access. Create extra blocks:
    # last_infall_idx = np.sum(~np.isnan(gas['infall_a']), axis=-1) - 1
    last_infall_idx = np.sum(gas['infall_a'] != invalid, axis=-1) - 1
    last_infall_idx = np.arange(len(last_infall_idx)), last_infall_idx
    for last, alle in [('last_infall_a', 'infall_a'),
                       ('last_infall_time', 'infall_time'),
                       ('mass_at_last_infall', 'mass_at_infall'),
                       ('metals_at_last_infall', 'metals_at_infall'),
                       ('jz_at_last_infall', 'jz_at_infall'),
                       ('T_at_last_infall', 'T_at_infall')]:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('  "%s"' % last)
        gas[last] = UnitArr(np.empty(len(gas), dtype=gas[alle].dtype),
                            units=gas[alle].units)
        gas[last][~trmask] = invalid
        gas[last][trmask] = gas[alle][last_infall_idx][trmask]
    del last_infall_idx
    environment.gc_full_collect()

    # last_ejection_idx = np.sum(~np.isnan(gas['ejection_a']), axis=-1) - 1
    last_ejection_idx = np.sum(gas['ejection_a'] != invalid, axis=-1) - 1
    last_ejection_idx = np.arange(len(last_ejection_idx)), last_ejection_idx
    for last, alle in [('last_ejection_a', 'ejection_a'),
                       ('last_ejection_time', 'ejection_time'),
                       ('mass_at_last_ejection', 'mass_at_ejection'),
                       ('metals_at_last_ejection', 'metals_at_ejection'),
                       ('jz_at_last_ejection', 'jz_at_ejection'),
                       ('T_at_last_ejection', 'T_at_ejection')]:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('  "%s"' % last)
        gas[last] = UnitArr(np.empty(len(gas), dtype=gas[alle].dtype),
                            units=gas[alle].units)
        gas[last][~trmask] = invalid
        gas[last][trmask] = gas[alle][last_ejection_idx][trmask]
    del last_ejection_idx
    environment.gc_full_collect()

    """
    inmask  = (gas['trace_type'] == 1)
    outmask = (gas['trace_type'] == 2)

    gididx_in = gididx[inmask[gididx]]
    gididx_out = gididx[outmask[gididx]]
    trididx_in = trididx[(trtype==1)[trididx]]
    trididx_out = trididx[(trtype==2)[trididx]]

    # Amount of metals gained nside and outside the galaxy.
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('  "metal_gain_in", "metal_gain_out"')
    # all traced particles have a value for metals gained before entering:
    gas['metal_gain_in'] = UnitArr( np.zeros(len(gas), dtype=float),
                                    units=units['MASS'] )
    gas['metal_gain_out'] = gas['metals_at_infall'][:,0].in_units_of(units['MASS'],copy=True)
    gas['metal_gain_in'][~trmask] = np.nan
    gas['metal_gain_out'][~trmask] = np.nan
    gas['metal_gain_in'][~trmask] = invalid
    gas['metal_gain_out'][~trmask] = invalid
    metals_last_enter = np.empty( len(data), dtype=float )
    metals_last_enter[trididx] = \
            gas['metals_at_infall'][:,0][gididx_traced].in_units_of(units['MASS'],copy=True)
    # some had gained more in (full) re-cycles (with no recycles: done!)
    for n in sorted(set_N_cycles)[1:]:  # 1,2,3,4,...
        gididx_n = gididx[(gas['num_recycled']==n)[gididx]]
        trididx_n = trididx[(n_cyc==n)[trididx]]
        # leaving the region
        idx = 2 + (n-1)*3
        metals_leave = np.array( [(i[idx][2] if i[0][1]>=n else invalid)
                                  for i in data.itervalues()] )
        idx = 1 + n*3
        metals_enter = np.array( [(i[idx][2] if i[0][1]>=n else invalid)
                                  for i in data.itervalues()] )
        gas['metal_gain_in'][gididx_n] += (metals_leave-metals_last_enter)[trididx_n]
        gas['metal_gain_out'][gididx_n] += (metals_enter-metals_leave)[trididx_n]
        metals_last_enter = metals_enter
    del gididx_n, trididx_n, metals_enter, metals_leave, metals_last_enter
    environment.gc_full_collect()
    gasmetals = gas['metals'].in_units_of(units['MASS'],subs=snap)
    # ... those that are currently outside, have gained something since they left
    # the region for the last time
    metals_last_leave = np.array( [(i[2+i[0][1]*3][2] if i[0][0]==2 else invalid)
                                   for i in data.itervalues()] )
    gas['metal_gain_out'][gididx_out] += gasmetals[gididx_out] - metals_last_leave[trididx_out]
    # ... and have gained something during they have been inside the region (which
    # was not a full cycle!)
    metals_last_enter = np.array( [i[1+i[0][1]*3][2] for i in data.itervalues()] )
    gas['metal_gain_in'][gididx_out] += \
            metals_last_leave[trididx_out] - metals_last_enter[trididx_out]
    # ... and those that are currently inside, have gained something since the
    # last time they entered
    gas['metal_gain_in'][gididx_in] += gasmetals[gididx_in] - metals_last_enter[trididx_in]
    del gasmetals, metals_last_enter, metals_last_leave
    environment.gc_full_collect()
    gas['metal_gain_out'].convert_to(gas['metals'].units)
    gas['metal_gain_in'].convert_to(gas['metals'].units)
    """


def read_pyigm_COS_halos_EWs(transitions=('HI 1215', 'MgII 2796', 'MgII 2803', 'CIV 1548',
                                          'CIV 1550', 'OVI 1031', 'OVI 1037')):
    '''
    Read the EW as a function of impact parameter of the COS halos using pyigm.

    References: Tumlinson+11; Werk+12; Tumlinson+13; Werk+13; Werk+14

    Args:
        transitions (iterable): The line transition to load the data for.
                                Eg. 'HI 1215' for Lyman-alpha. Or 'HI 1025',
                                'MgII 2796', 'MgII 2803', 'OVI 1037', 'OVI 1031',
                                'SiIII 1206', 'CIII 977', 'CIV 1548', 'CIV 1550',
                                etc.

    Returns:
        data (dict):            Relevant data for each line of sight. It includes:
                                'z' (reshifts), 'rho' (impact parameter),
                                'EW' (equivalent widths of the lines),
                                'Mstars' (stellar masses of the associated
                                galaxies), and more. Each entry is a UnitArr.
    '''
    import pyigm
    from pyigm.cgm import cos_halos
    cos_halos = cos_halos.COSHalos()
    data = {trans: [] for trans in transitions}
    for sys in cos_halos:
        sys = sys.to_dict()
        Mhalo = sys['galaxy']['halo_mass']
        Mstars = sys['galaxy']['stellar_mass']
        sSFR = sys['galaxy']['ssfr']
        Rvir = sys['galaxy']['rvir']
        for component in sys['igm_sys']['components']:
            for line in sys['igm_sys']['components'][component]['lines']:
                tname = sys['igm_sys']['components'][component]['lines'][line]['name']
                if tname in transitions:
                    EW = sys['igm_sys']['components'][component]['lines'][line]['attrib']['EW']['value']
                    data[tname].append([sys['z'], sys['rho'], EW,
                                        Mhalo, Mstars, sSFR, Rvir])
    for trans, d in data.items():
        if len(d) == 0:
            raise RuntimeError('transition "%s" not found' % trans)

    data = {trans: np.array(d) for trans, d in data.items()}
    return {trans: {
        'z': UnitArr(d[:, 0]),
        'rho': UnitArr(d[:, 1], 'kpc'),
        # 'EW':       UnitArr(10.**d[:,2], 'Angstrom'),
        'EW': UnitArr(d[:, 2], 'Angstrom'),
        'Mhalo': UnitArr(10. ** d[:, 3], 'Msol'),
        'Mstars': UnitArr(10. ** d[:, 4], 'Msol'),
        'sSFR': UnitArr(d[:, 5]),
        'Rvir': UnitArr(d[:, 6], 'kpc'),
    } for trans, d in data.items()}

