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

def prepare_zoom(s, info='deduce', fullsph=False):
    '''
    A convenience function to load a snapshot from a zoomed-in simulation that is
    not yet centered or orienated.

    Args:
        s (str, Snap):      The snapshot of a zoomed-in simulation to prepare.
                            Either as an already loaded snapshot or a path to the
                            snapshot.
        info (str, dict):   Path to info file or the dictionary as returned from
                            `read_info_file`.
                            However, if set to 'deduce', it is tried to deduce the
                            path to the info file from the snapshot filename: it
                            is assumed to be in a subfolder 'trace' that is in the
                            same directory as the snapshot and named
                            `info_%03d.txt`, where the `%03d` is filled with the
                            snapshot number. The latter is taken as the last three
                            characters of the first dot / the end of the filename.
        fullsph (bool):     Whether to mask all particles that overlap into the
                            halo region (that is also include SPH particles that
                            are outside the virial radius, but their smoothing
                            length reaches into it).

    Returns:
        s (Snap):           The prepared snapshot.
        halo (SubSnap):     The cut halo of the found structure.
    '''
    if isinstance(s,str):
        s = Snap(s)
    print 'prepare zoomed-in', s

    if info is 'deduce':
        try:
            snap = int(os.path.basename(s.filename).split('.')[0][-3:])
            info = os.path.dirname(s.filename) + '/trace/info_%03d.txt' % snap
        except:
            info = None
    if isinstance(info, str):
        if not os.path.exists(info):
            print >> sys.stderr, 'WARNING: There is not info file named ' + \
                                 '"%s"' % info
            info = None
        else:
            info = read_info_file(info)

    s.to_physical_units()

    # center in space
    if info:
        center = info['center']
    else:
        center = shrinking_sphere(s.stars,
                                  [float(s.boxsize)/2.]*3,
                                  np.sqrt(3)*s.boxsize)
    print 'center at:', center
    Translation(-center).apply(s)
    # center the velocities
    s.vel -= mass_weighted_mean(s[s.r<'1 kpc'], 'vel')
    # orientate at the angular momentum of the baryons wihtin 10 kpc
    if info:
        L = info['L_baryons']
        orientate_at(s, 'vec', L, total=True)
    else:
        orientate_at(s[s.r < '10 kpc'].baryons, 'L', total=True)

    # cut the halo (<R200)
    if info:
        R200 = info['R200']
        M200 = info['M200']
    else:
        R200, M200 = virial_info(s)
    print 'R200:', R200
    print 'M200:', M200
    halo = s[BallMask(R200, fullsph=fullsph)]

    # cut the inner part (< 15% R200)
    gal = s[BallMask(0.15*R200, fullsph=fullsph)]
    Ms = gal.stars.mass.sum()
    print 'M*:  ', Ms

    return s, halo

def fill_star_from_info(snap, SFI):
    '''
    Read the formation radius rform and rform/R200(aform) from the star_form.ascii
    file and create the blocks rform and rR200form.

    Args:
        snap (Snap):    The snapshot to fill with the data (has to be the one at
                        z=0 of the simulation used to create the star_form.ascii).
        SFI (str):      The path to the star_form.ascii file.
    '''
    stars = snap.root.stars
    SFI = np.loadtxt(SFI, skiprows=1)
    if set(stars.ID) != set(SFI[:,0]):
        raise RuntimeError('Stellar IDs are not the exactly those from ' + \
                           '"%s"!' % SFI)

    sfiididx = np.argsort( SFI[:,0] )
    sididx = np.argsort( stars.ID )

    stars.add_custom_block(UnitArr(np.empty(len(stars),dtype=float),units='kpc'),
                           'rform')
    stars.rform[sididx] = SFI[:,2][sfiididx]

    stars.add_custom_block(UnitArr(np.empty(len(stars),dtype=float)),
                           'rR200form')
    stars.rR200form[sididx] = SFI[:,3][sfiididx]

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

