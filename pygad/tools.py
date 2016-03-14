'''
Some general high-level functions.

Example:
'''
__all__ = ['read_info_file', 'prepare_zoom', 'fill_star_from_info',
           'read_traced_gas']

from snapshot import *
from units import *
from analysis import *
from transformation import *
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
    with open(filename, 'r') as finfo:
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
                 sph_overlap_mask=False, gal_R200=0.10, **kwargs):
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
                            "FoF" (no reasonable default, hence the default to
                            None).
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

    Returns:
        s (Snap):           The prepared snapshot.
        halo (SubSnap):     The cut halo of the found structure.
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
    print 'prepare zoomed-in', s

    # read info file (if required)
    if mode in ['auto', 'info']:
        if info is 'deduce':
            try:
                snap = int(os.path.basename(s.filename).split('.')[0][-3:])
                info = os.path.dirname(s.filename) + '/trace/info_%03d.txt' % snap
            except:
                info = None
        if isinstance(info, str):
            if not os.path.exists(info):
                print >> sys.stderr, 'WARNING: There is no info file named ' + \
                                     '"%s"' % info
                info = None
            else:
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

    s.to_physical_units()

    # find center
    if mode == 'info':
        center = info['center']
    elif mode in ['ssc', 'FoF']:
        if mode == 'FoF':
            #raise NotImplementedError('Mode "FoF" is not yet implemented.')
            if linking_length is None:
                raise ValueError('You have to define `linking_length` in ' +
                                 '"FoF" mode -- cannot be None!')
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
                # The most massive galaxy does not have to be in a halo with litle low
                # resolution elements! Find the most massive galaxy living in a
                # "resolved" halo:
                galaxy = None
                for gal in galaxies:
                    # since the same linking lengths were used for the halos and the
                    # galaxies (rethink that!), a galaxy in a halo is entirely in that
                    # halo or not at all
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
            from utils import periodic_distance_to
            R = np.max(periodic_distance_to(s['pos'], com, s.boxsize))
            center = shrinking_sphere(shrink_on, com, R)
        else:
            center = None
    else:
        raise ValueError('Unkown mode "%s"!' % mode)

    # center in space
    if center is None:
        print 'no center found -- do not center'
    else:
        print 'center at:', center
        Translation(-center).apply(s)
        # center the velocities
        vel_center = mass_weighted_mean(s[s['r']<'1 kpc'], 'vel')
        print 'center velocities at:', vel_center
        s['vel'] -= vel_center

    # cut the halo (<R200)
    if mode == 'info':
        R200 = info['R200']
        M200 = info['M200']
    else:
        print 'derive virial information'
        R200, M200 = virial_info(s)
    print 'R200:', R200
    print 'M200:', M200
    halo = s[BallMask(R200, sph_overlap=sph_overlap_mask)]

    # orientate at the reduced inertia tensor of the baryons wihtin 10 kpc
    print 'orientate',
    if mode == 'info':
        if 'I_red(gal)' in info:
            redI = info['I_red(gal)']
            if redI is not None:
                redI = redI.reshape((3,3))
            print 'at the galactic red. inertia tensor from info file:'
            print redI
            mode, qty = 'red I', redI
        else:
            print 'at angular momentum of the galaxtic baryons from info file:'
            mode, qty = 'vec', info['L_baryons']
        orientate_at(s, mode, qty=qty, total=True)
    else:
        print 'at red. inertia tensor of the baryons within %.3f*R200' % gal_R200
        orientate_at(s[BallMask(gal_R200*R200, sph_overlap=False)].baryons,
                     'red I',
                     total=True
        )

    # cut the inner part as the galaxy
    gal = s[BallMask(gal_R200*R200, sph_overlap=sph_overlap_mask)]
    Ms = gal.stars['mass'].sum()
    print 'M*:  ', Ms

    if len(gal)==0:
        gal = None
    if len(halo)==0:
        halo = None

    if mode=='FoF' and ret_FoF:
        return s, halo, gal, halos
    else:
        return s, halo, gal

def fill_star_from_info(snap, filename):
    '''
    Read the formation radius rform and rform/R200(aform) from the star_form.ascii
    file and create the new blocks "rform" and "rR200form".

    Note:
        The set of the star particle IDs in the star formation information file
        must exactly be the set of star particle IDs in the (root) snapshot.

    Args:
        snap (Snap):    The snapshot to fill with the data (has to be the one at
                        z=0 of the simulation used to create the star_form.ascii).
        filename (str): The path to the star_form.ascii file.
    '''
    stars = snap.root.stars
    if environment.verbose:
        print 'reading the star formation information from %s...' % filename
    SFI = np.loadtxt(filename, skiprows=1)
    if environment.verbose:
        print 'testing if the IDs math the (root) snapshot...'
    SFI_IDs = SFI[:,0].astype(int)
    if set(stars['ID']) != set(SFI_IDs) or len(SFI_IDs) != len(stars):
        raise RuntimeError('Stellar IDs do not exactly match those from ' + \
                           '"%s"!' % filename)

    if environment.verbose:
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
        1:  gas in disc
        2:  gas out of disc
        3:  stars that formed in disc
        4:  traced gas, that formed stars outside disc
    and the number of full cycles (leaving disc + re-entering).

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
                                  events of leaving + out, forming a star, or
                                  leaving + out + forming a stars, depending on
                                  type
                                ]
                            where n is the number of full recylces.
    '''
    if environment.verbose:
        print 'read file...'
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
    if environment.verbose:
        print 'restructure by type...'
        sys.stdout.flush()
    # structure the data into sub-lists
    # (re-)enter:   5 elements
    # leave:        6 elements
    # star form:    4 elements
    # being out:    4 elements
    # -> full recycle:  6+4+5 = 15
    # -> in disc:       5 + n*15                =  5 + n*15
    #    out:           5 + n*15 + 6 + 4        = 15 + n*15
    #    SF in disc:    5 + n*15 + 4            =  9 + n*15
    #    SF outside:    5 + n*15 + 6 + 4 + 4    = 19 + n*15
    t_to_type = {5:1, 15:2, 9:3, 19:4}
    for ID in tr.keys():
        e = tr[ID]
        t = len(e) % 15
        if t in [0,4]: t += 15
        tt = t_to_type[t]
        if tt not in types:
            del tr[ID]
            continue
        sub = e[5:{5:None,15:-10,9:-4,19:-14}[t]]
        new = [[tt, len(sub)/15]]
        new += [e[:5]]
        for re in np.array(sub).reshape(((len(e)-t)/15,15)):
            new += [re[:6]]
            new += [re[6:10]]
            new += [re[10:]]
        if t == 5:      # gas in disc (with possible recycles)
            pass
        elif t == 9:    # turned in the disc into a star
            new += [e[-4:]]
        elif t == 15:   # left disc, but is still a gas particle
            new += [e[-10:-4],e[-4:]]
        elif t == 19:   # left disc and turned into a star
            new += [e[-14:-8],e[-8:-4],e[-4]]
        else:
            raise RuntimeError('Structure in "%s" ' % filename + \
                               'is not as expected!')
        tr[ID] = new

    return tr

