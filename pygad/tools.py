'''
Some general high-level functions.

Example:
'''
__all__ = ['read_info_file', 'prepare_zoom', 'fill_star_from_info',
           'read_traced_gas', 'fill_gas_from_traced',
           'fill_derived_gas_trace_qty']

from snapshot import *
from units import *
from utils import *
from analysis import *
from transformation import *
import gadget
import environment
import re
import sys
import os

def read_info_file(filename):
    '''
    Read in the contents of an info file as produced by gtrace into a dictionary.

    It is assumed, that there is excatly one colon per line, which seperates the
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
                name, value = line.split(':', 1)
                name, value = name.strip(), value.strip()
                blocks = re.findall('\[(.*?)\]', value)
                if len(blocks) == 0:
                    value = float(value) if value != 'None' else None
                elif len(blocks) == 1:
                    value = UnitArr(float(value.rsplit('[')[0].strip()),
                                    units=blocks[0])
                elif len(blocks) == 2:
                    value = UnitArr(map(lambda s: float(s.strip(', ')),
                                        blocks[0].split()),
                                    units=None if blocks[1]=='None' \
                                            else blocks[1])
                if name in info:
                    print >> sys.stderr, 'WARNING: "%s" occures ' % name + \
                                         'multiple times in info file ' + \
                                         '"%s"! First one used.' & filename
                info[name] = value
            except ValueError as e:
                if e.message[-6:] != 'unpack':
                    raise
    return info

def prepare_zoom(s, mode='auto', info='deduce', shrink_on='stars',
                 linking_length=None, linking_vel='200 km/s', ret_FoF=False,
                 sph_overlap_mask=False, gal_R200=0.10, star_form='deduce',
                 gas_trace='deduce', to_physical=True, **kwargs):
    '''
    A convenience function to load a snapshot from a zoomed-in simulation that is
    not yet centered or orienated.

    Args:
        s (str, Snap):      The snapshot of a zoomed-in simulation to prepare.
                            Either as an already loaded snapshot or a path to the
                            snapshot.
        mode (str):         The mode in which to prepare the snapshot. You can
                            choose from:
                                * 'auto':   try 'info', if it does not work
                                            fallback to 'ssc'
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

    Returns:
        s (Snap):           The prepared snapshot.
        halo (SubSnap):     The cut halo of the found structure.
        gal (SubSnap):      The central galaxy of the halo as defined by
                            `gal_R200`.
    '''
    def get_shrink_on_sub(snap):
        if isinstance(shrink_on, str):
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
        return M_lowres/M > threshold
            

    if isinstance(s,str):
        s = Snap(s)
    gal_R200 = float(gal_R200)
    if environment.verbose >= environment.VERBOSE_TACITURN:
        print 'prepare zoomed-in', s

    # read info file (if required)
    if mode in ['auto', 'info']:
        if info is 'deduce':
            try:
                snap = int(os.path.basename(s.filename).split('.')[0][-3:])
                info = os.path.dirname(s.filename) + '/trace/info_%03d.txt' % snap
            except:
                print >> sys.stderr, 'WARNING: could not deduce the path to ' + \
                                     'the trace file!'
                info = None
        if isinstance(info, str):
            info = os.path.expanduser(info)
            if not os.path.exists(info):
                print >> sys.stderr, 'WARNING: There is no info file named ' + \
                                     '"%s"' % info
                info = None
            else:
                if environment.verbose >= environment.VERBOSE_TACITURN:
                    print 'read info file from:', info
                info = read_info_file(info)
        if info is None:
            if mode == 'auto':
                mode = 'ssc'
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
                    l = linking_length,
                    exclude = FoF_exclude,
                    calc = ['mass', 'lowres_mass'],
                    max_halos = 10,
                    **kwargs
            )
            if shrink_on not in ['all', 'highres']:
                galaxies = generate_FoF_catalogue(
                        get_shrink_on_sub(s),
                        l = linking_length,
                        dvmax = linking_vel,
                        calc = ['mass', 'com'],
                        max_halos = 10,
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
                shrink_on = s[galaxy]
            else:
                shrink_on = s[halos[0]]
        elif mode == 'ssc':
            shrink_on = get_shrink_on_sub(s)
        if shrink_on is not None and len(shrink_on)>0:
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
            print 'no center found -- do not center'
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'center at:', center
        Translation(-center).apply(s)
        # center the velocities
        vel_center = mass_weighted_mean(s[s['r']<'1 kpc'], 'vel')
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'center velocities at:', vel_center
        s['vel'] -= vel_center

    # cut the halo (<R200)
    if mode == 'info':
        R200 = info['R200']
        M200 = info['M200']
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'derive virial information'
        R200, M200 = virial_info(s)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'R200:', R200
        print 'M200:', M200
    halo = s[BallMask(R200, sph_overlap=sph_overlap_mask)]

    # orientate at the reduced inertia tensor of the baryons wihtin 10 kpc
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'orientate',
    if mode == 'info':
        if 'I_red(gal)' in info:
            redI = info['I_red(gal)']
            if redI is not None:
                redI = redI.reshape((3,3))
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print 'at the galactic red. inertia tensor from info file'
            if environment.verbose >= environment.VERBOSE_TALKY:
                print redI
            mode, qty = 'red I', redI
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print 'at angular momentum of the galaxtic baryons from info file:'
            mode, qty = 'vec', info['L_baryons']
        orientate_at(s, mode, qty=qty, total=True)
    else:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'at red. inertia tensor of the baryons within %.3f*R200' % gal_R200
        orientate_at(s[BallMask(gal_R200*R200, sph_overlap=False)].baryons,
                     'red I',
                     total=True
        )

    # cut the inner part as the galaxy
    gal = s[BallMask(gal_R200*R200, sph_overlap=sph_overlap_mask)]
    Ms = gal.stars['mass'].sum()
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'M*:  ', Ms

    if len(gal)==0:
        gal = None
    if len(halo)==0:
        halo = None

    if star_form == 'deduce':
        try:
            star_form = os.path.dirname(s.filename) + '/trace/star_form.ascii'
        except:
            print >> sys.stderr, 'WARNING: could not deduce the path to the ' + \
                                 'star formation file!'
            star_form = None
    if isinstance(star_form, str):
        star_form = os.path.expanduser(star_form)
        if not os.path.exists(star_form):
            print >> sys.stderr, 'WARNING: There is no star formation file ' + \
                                 'named "%s"' % star_form
            star_form = None
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print 'read star formation file from:', star_form
            fill_star_from_info(s, star_form)

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
                raise RuntimeError
        except:
            print >> sys.stderr, 'WARNING: could not deduce the path to the ' + \
                                 'gas tracing file!'
            gas_trace = None
    if isinstance(gas_trace, str):
        gas_trace = os.path.expanduser(gas_trace)
        if not os.path.exists(gas_trace):
            print >> sys.stderr, 'WARNING: There is no gas trace file named ' + \
                                 '"%s"' % gas_trace
            gas_trace = None
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print 'read gas trace file from:', gas_trace
            fill_gas_from_traced(s, gas_trace)

    if mode=='FoF' and ret_FoF:
        return s, halo, gal, halos
    else:
        return s, halo, gal

def fill_star_from_info(snap, data):
    '''
    Read the formation radius rform and rform/R200(aform) from the star_form.ascii
    file and create the new blocks "rform" and "rR200form".

    Note:
        The set of the star particle IDs in the star formation information file
        must exactly be the set of star particle IDs in the (root) snapshot.

    Args:
        snap (Snap):    The snapshot to fill with the data (has to be the one at
                        z=0 of the simulation used to create the star_form.ascii).
        data (str, np.ndarray): 
                        The path to the star_form.ascii file or the already
                        read-in data.
    '''
    stars = snap.root.stars

    if isinstance(data, str):
        filename = data
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print 'reading the star formation information from %s...' % filename
        SFI = np.loadtxt(filename, skiprows=1)
    else:
        filename = '<data given>'
        SFI = data

    if environment.verbose >= environment.VERBOSE_TALKY:
        print 'testing if the IDs match the (root) snapshot...'
    SFI_IDs = SFI[:,0].astype(int)
    if set(stars['ID']) != set(SFI_IDs) or len(SFI_IDs) != len(stars):
        raise RuntimeError('Stellar IDs do not exactly match those from ' + \
                           '"%s" (mismatch: %d/%d)!' % (filename,
                               len(set(stars['ID'])-set(SFI_IDs)),
                               len(set(SFI_IDs)-set(stars['ID']))))

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'adding the new blocks "rform" and "rR200form"...'
    sfiididx = np.argsort( SFI_IDs )
    sididx = np.argsort( stars['ID'] )

    stars['rform'] = UnitArr( np.empty(len(stars),dtype=float), units='kpc' )
    stars['rform'][sididx] = SFI[:,2][sfiididx]

    stars['rR200form'] = np.empty(len(stars),dtype=float)
    stars['rR200form'][sididx] = SFI[:,3][sfiididx]

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
        print 'read gas trace file "%s"...' % filename
        sys.stdout.flush()
    import cPickle as pickle
    with file(filename, 'rb') as f:
        tr = pickle.load(f)

    #sort type
    if types is None or types=='all':
        types = set([1,2,3,4])
    else:
        if isinstance(types,int):
            types = [types]
        types = set(types)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print 'restructure by type...'
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
    t_to_type = {5:1, 15:2, 9:3, 19:4}
    for ID in tr.keys():
        e = tr[ID]
        t = len(e) % 15
        if t in [0,4]: t += 15
        tt = t_to_type[t]
        if tt not in types:
            print >> sys.stderr, 'ERROR: could not process particle ID %d' % ID
            print >> sys.stderr, '       skip: %s' % e
            del tr[ID]
            continue
        sub = e[ 5 : {5:None,15:-10,9:-4,19:-14}[t] ]
        #assert len(sub) % 15 == 0
        new = [[tt, len(sub)/15]]
        new += [e[:5]]
        for re in np.array(sub).reshape(((len(e)-t)/15,15)):
            new += [re[:6]]
            new += [re[6:10]]
            new += [re[10:]]
        if t == 5:      # gas in region (with possible recycles)
            pass
        elif t == 9:    # turned in the region into a star
            new += [e[-4:]]
        elif t == 15:   # left region, but is still a gas particle
            new += [e[-10:-4],e[-4:]]
        elif t == 19:   # left region and turned into a star
            new += [e[-14:-8],e[-8:-4],e[-4]]
        else:
            raise RuntimeError('Structure in "%s" ' % filename + \
                               'is not as expected!')
        tr[ID] = new

    return tr

def fill_gas_from_traced(snap, data, add_derived=True,
                         #TODO: angmom units
                         units=dict(TIME='a_form', MASS='Msol', TEMP='K',
                                    ANGMOM=None, POS='kpc'),
                         invalid=0.0):
    '''
    Fill some information from the gas trace file into the snapshot as blocks.

    This function is using data from a gas trace file (as produced by `gtracegas`)
    to create new gas blocks with particle properties at in- and outflow times as
    well as additional information of the time being ejected.

    Among the new blocks are:
      "trace_type", "num_recycled",
      "infall_a", "infall_time", "mass_at_infall", "metals_at_infall", "jz_at_infall", "T_at_infall",
      "ejection_a", "mass_at_ejection"

    Note:
        The set of the star particle IDs in the star formation information file
        must exactly be the set of star particle IDs in the (root) snapshot.

    Args:
        snap (Snap):        The snapshot to fill with the data (has to be the one
                            at z=0 of the simulation used to create the trace
                            file).
        data (str, dict):   The path to the gas trace file or the alread read-in
                            data.
        add_derived (bool): Whether to also add blocks that can be derived from
                            the trace data stored in blocks. The function
                            `fill_derived_gas_trace_qty` is used for that.
        units (dict):       The units to use for masses, lengths, etc..
        invalid (bool):     The value to fill invalid entries with. Such entries
                            are properties of cycles that did not happen for a
                            given particle.
    '''
    gas = snap.root.gas

    if isinstance(data,str):
        filename = data
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print 'reading the gas trace information from %s...' % filename
        data = read_traced_gas(data)
    else:
        filename = '<given data>'

    # filter to gas only
    gas_type = set([1,2])
    data = dict( filter(lambda i:i[1][0][0] in gas_type,
                        data.iteritems() ) )

    if environment.verbose >= environment.VERBOSE_TALKY:
        print 'test IDs and find matching IDs...'
    if len(set(data.keys())-set(gas['ID'])) > 0:
        raise RuntimeError('Traced gas IDs in "%s" have ' % filename +
                           '%s ' % nice_big_num_str(len(
                               set(data.keys())-set(gas['ID']))) +
                           'elements that are not in the snapshot!')
    tracedIDs = (set(data.keys()) & set(gas['ID']))

    trmask = np.array( [(ID in tracedIDs) for ID in gas['ID']], dtype=bool)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print '  found %s (of %s)' % (nice_big_num_str(len(tracedIDs)),
                                      nice_big_num_str(len(data))),
        print 'traced IDs that are in the snapshot'

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'adding the blocks:'

    trididx = np.argsort( data.keys() )
    gididx = np.argsort( gas['ID'] )
    gididx_traced = gididx[trmask[gididx]]

    # type: not traced (0), in region (1), out of region (2)
    # number of full cycles (out and in)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "trace_type", "num_recycled",'

    gas['trace_type'] = UnitArr( np.zeros(len(gas), dtype=int) )
    trtype = np.array( [i[0][0] for i in data.itervalues()] )
    gas['trace_type'][gididx_traced] = trtype[trididx]

    inmask  = (gas['trace_type'] == 1)
    outmask = (gas['trace_type'] == 2)
    gididx_in = gididx[inmask[gididx]]
    gididx_out = gididx[outmask[gididx]]
    trididx_in = trididx[(trtype==1)[trididx]]
    trididx_out = trididx[(trtype==2)[trididx]]

    n_cyc = np.array( [i[0][1] for i in data.itervalues()] )
    gas['num_recycled'] = UnitArr( np.empty(len(gas), dtype=n_cyc.dtype) )
    gas['num_recycled'][~trmask] = -1
    gas['num_recycled'][gididx_traced] = n_cyc[trididx]
    set_N_cycles = set(n_cyc)
    if environment.verbose >= environment.VERBOSE_TALKY:
        print '  +++ number of recycles that occured:', set_N_cycles, '+++'


    # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
    # events of entering the region. N is the number of the gas particles. Each
    # particle has entries for each of their enter eventss, those events that do
    # not exists, get filled with `invalid`.
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "infall_a", "infall_time", "mass_at_infall", ' + \
              '"metals_at_infall", "jz_at_infall", "T_at_infall",'

    max_N_cycle = max(set_N_cycles)
    infall_t = np.array( [[e[0] for e in i[1:1+3*n+1:3]] + [invalid]*(max_N_cycle-n)
                          for n,i in zip(n_cyc, data.values())] )
    gas['infall_a'] = UnitArr( np.empty((len(gas),max_N_cycle+1),
                                      dtype=infall_t.dtype), units=units['TIME'] )
    gas['infall_a'][~trmask] = invalid
    gas['infall_a'][gididx_traced] = infall_t[trididx]
    del infall_t

    from snapshot import age_from_form
    # only convert reasonable values & ensure not to overwrite blocks
    mask = (gas['infall_a']!=invalid) & np.isfinite(gas['infall_a'])
    gas['infall_time'] = gas['infall_a'].copy()
    new = gas.cosmic_time() - age_from_form(gas['infall_time'][mask],subs=gas)
    gas['infall_time'].units = new.units
    gas['infall_time'][mask] = new
    del new

    for name,idx,unit in [('mass_at_infall',   1, units['MASS']),
                          ('metals_at_infall', 2, units['MASS']),
                          ('jz_at_infall',     3, units['ANGMOM']),
                          ('T_at_infall',      4, units['TEMP'])]:
        infall_Q = np.array( [[e[idx] for e in i[1:1+3*n+1:3]] +
                                    [invalid]*(max_N_cycle-n)
                              for n,i in zip(n_cyc, data.values())] )
        gas[name] = UnitArr( np.empty((len(gas),max_N_cycle+1),
                                      dtype=infall_Q.dtype),
                             units=unit )
        gas[name][~trmask] = invalid
        gas[name][gididx_traced] = infall_Q[trididx]
        del infall_Q


    # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
    # events of ejection / leaving the region. N is the number of the gas
    # particles. Each particle has entries for each of their ejection events,
    # those events that do not exists, get filled with `invalid`.
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "ejection_a", "ejection_time", "mass_at_ejection", ' + \
              '"metals_at_ejection", "jz_at_ejection", "T_at_ejection",'

    max_N_cycle = max(set_N_cycles)
    eject_t = np.array( [[e[0] for e in i[2:2+3*n:3]] +
                            [i[2+3*n][0] if t==2 else invalid] +
                            [invalid]*(max_N_cycle-n)
                         for n,t,i in zip(n_cyc, trtype, data.values())] )
    gas['ejection_a'] = UnitArr( np.empty((len(gas),max_N_cycle+1),
                                          dtype=eject_t.dtype), units=units['TIME'] )
    gas['ejection_a'][~trmask] = invalid
    gas['ejection_a'][gididx_traced] = eject_t[trididx]
    del eject_t

    from snapshot import age_from_form
    # only convert reasonable values & ensure not to overwrite blocks
    mask = (gas['ejection_a']!=invalid) & np.isfinite(gas['ejection_a'])
    gas['ejection_time'] = gas['ejection_a'].copy()
    new = gas.cosmic_time() - age_from_form(gas['ejection_time'][mask],subs=gas)
    gas['ejection_time'].units = new.units
    gas['ejection_time'][mask] = new
    del new


    for name,idx,unit in [('mass_at_ejection',   1, units['MASS']),
                          ('metals_at_ejection', 2, units['MASS']),
                          ('jz_at_ejection',     3, units['ANGMOM']),
                          ('T_at_ejection',      4, units['TEMP'])]:
        eject_Q = np.array( [[e[idx] for e in i[2:2+3*n:3]] +
                                [i[2+3*n][idx] if t==2 else invalid] +
                                [invalid]*(max_N_cycle-n)
                             for n,t,i in zip(n_cyc, trtype, data.values())] )
        gas[name] = UnitArr( np.empty((len(gas),max_N_cycle+1),
                                      dtype=eject_Q.dtype),
                             units=unit )
        gas[name][~trmask] = invalid
        gas[name][gididx_traced] = eject_Q[trididx]
        del eject_Q


    # for each cycle ther is a maximum travel distance, plus one more for those
    # particles that are outside the region: store them
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "cycle_r_max", "cycle_z_max"'
    for name,idx,unit in [('cycle_r_max', 1, units['POS']),
                          ('cycle_z_max', 3, units['POS'])]:
        pos = np.array( [[e[idx] for e in i[3:3+3*n:3]] +
                            [i[3+3*n][idx] if t==2 else invalid] +
                            [invalid]*(max_N_cycle-n)
                         for n,t,i in zip(n_cyc, trtype, data.values())] )
        gas[name] = UnitArr( np.empty((len(gas),max_N_cycle+1),
                                      dtype=pos.dtype),
                             units=unit )
        gas[name][~trmask] = invalid
        gas[name][gididx_traced] = pos[trididx]
        del pos

    environment.gc_full_collect()
    if add_derived:
        fill_derived_gas_trace_qty(snap, units=units, invalid=invalid)


def fill_derived_gas_trace_qty(snap,
                               #TODO: angmom units
                               units=dict(TIME='a_form', MASS='Msol', TEMP='K',
                                          ANGMOM=None, POS='kpc'),
                               invalid=0.0):
    """
    Derive blocks from the gas trace blocks and add them.

    Among the new blocks are:
      "cycle_r_max", "cycle_z_max"
      "out_time"
      "last_infall_a",
      "last_ejection_a", "last_ejection_time", "mass_at_last_ejection"

    Note:
        The arguments `units` and `invalid` shall be the same as for
        `fill_gas_from_traced`.

    Args:
        snap (Snap):        The snapshot to fill with the data (has to be the one
                            at z=0 of the simulation used to create the trace
                            file).
        units (dict):       The units to use for masses, lengths, etc..
        invalid (bool):     The value to fill invalid entries with. Such entries
                            are properties of cycles that did not happen for a
                            given particle.
    """
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'adding blocks that can be derived from the gas trace blocks:'

    gas = snap.gas

    trmask = (gas['num_recycled'] != -1)
    set_N_cycles = set(gas['num_recycled'])
    max_N_cycle = max(set_N_cycles)

    # each (full) cycle takes some given time
    if max_N_cycle>0:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print '  "out_time",'
        ejected = gas['ejection_time'][:,:-1]
        infall = gas['infall_time'][:,1:]
        gas['out_time'] = infall - ejected
        mask = (ejected==invalid) | ~np.isfinite(ejected)
        del ejected
        environment.gc_full_collect()
        gas['out_time'][mask] = invalid
        mask = (infall==invalid) | ~np.isfinite(infall)
        del infall
        environment.gc_full_collect()
        gas['out_time'][mask] = invalid


    # The events of the last infall and the last ejection are a bit messy to
    # access. Create extra blocks:
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "last_infall_a", "last_infall_time"", "mass_at_last_infall",',
        print '"metals_at_last_infall", "jz_at_last_infall", "T_at_last_infall",'

    last_infall_idx = np.sum(~np.isnan(gas['infall_a']),axis=-1) - 1
    last_infall_idx = np.arange(len(last_infall_idx)),last_infall_idx
    for last,alle in [('last_infall_a', 'infall_a'),
                      ('last_infall_time', 'infall_time'),
                      ('mass_at_last_infall', 'mass_at_infall'),
                      ('metals_at_last_infall', 'metals_at_infall'),
                      ('jz_at_last_infall', 'jz_at_infall'),
                      ('T_at_last_infall', 'T_at_infall')]:
        gas[last] = UnitArr( np.empty(len(gas), dtype=gas[alle].dtype),
                                      units=gas[alle].units )
        gas[last][~trmask] = invalid
        gas[last][trmask] = gas[alle][last_infall_idx][trmask]
    del last_infall_idx

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "last_ejection_a", "last_ejection_time",',
        print '"mass_at_last_ejection", "metals_at_last_ejection",',
        print '"jz_at_last_ejection", "T_at_last_ejection"'

    last_ejection_idx = np.sum(~np.isnan(gas['ejection_a']),axis=-1) - 1
    last_ejection_idx = np.arange(len(last_ejection_idx)),last_ejection_idx
    for last,alle in [('last_ejection_a', 'ejection_a'),
                      ('last_ejection_time', 'ejection_time'),
                      ('mass_at_last_ejection', 'mass_at_ejection'),
                      ('metals_at_last_ejection', 'metals_at_ejection'),
                      ('jz_at_last_ejection', 'jz_at_ejection'),
                      ('T_at_last_ejection', 'T_at_ejection')]:
        gas[last] = UnitArr( np.empty(len(gas), dtype=gas[alle].dtype),
                                      units=gas[alle].units )
        gas[last][~trmask] = invalid
        gas[last][trmask] = gas[alle][last_ejection_idx][trmask]
    del last_ejection_idx


    """
    # Amount of metals gained nside and outside the galaxy.
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '  "metal_gain_in", "metal_gain_out"'
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
    gas['metal_gain_out'].convert_to(gas['metals'].units)
    gas['metal_gain_in'].convert_to(gas['metals'].units)
    """

