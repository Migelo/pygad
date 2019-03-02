import argparse

# # minimum number of particles required in the halos/galaxies
# min_N = 100
# # only find/return the most massive halos in the catalogues (5 should be
# # sufficient for most cases)
# max_halos = 1
# disc_def = dict() #jzjc_min=0.85, rmax='50 kpc', zmax='10 kpc')

RESTART_FILENAME = 'restart.gcache3'

parser = argparse.ArgumentParser(
            description='Trace the most massive progenitor of the most massive '
                        '(in stellar mass) galaxy in a highly resolved halo in '
                        'the start snapshot, that has no or very little low '
                        'resolution particles/mass, though the simulation. Store '
                        'its properties as well as storing the corresponding cut '
                        'region (centered at the center of the stars of the '
                        'galaxy of interest using a shrinking sphere and its '
                        'mass-weighted velocity).')
parser.add_argument('name_pattern',
                    help='The pattern of the snapshots\' file names. One ' +
                         'can use C- / Python-like string formatter that ' +
                         'are all going to be replaced by the snapshot ' +
                         'number. (example: "output/snap_%%03d.gdt")')
parser.add_argument('--command', '-c',
                    metavar='str',
                    default='',
                    help='the path+name of the script to process the snapshots (default: "" ')
parser.add_argument('--profile', '-p',
                    metavar='str',
                    default='cache',
                    help='the profile used to process the snapshots (default: "cache" ')
parser.add_argument('--withplot',
                    action='store_true',
                    help='create a plot in the cache.')

parser.add_argument('--destination', '-d',
                    metavar='FOLDER',
                    default=None,
                    help='The folder to store the output (default: <profile name> ' +
                         'in the directory of the snapshots).')
parser.add_argument('--start', '-s',
                    metavar='INT',
                    type=int,
                    default=None,
                    help='The number of the snapshot to start with (the ' +
                         'latest one). If there is a file called ' +
                         '"%s" in the destination directory,' % RESTART_FILENAME +
                         'restart the last run and ignore the start parameter. ' +
                         'By default it is looked for the highest consecutive ' +
                         'number in the snapshot filename pattern for which a ' +
                         'snapshot exists.')
parser.add_argument('--end', '-e',
                    metavar='INT',
                    type=int,
                    default=None,
                    help='The number of the logysnapshot to stop at (the ' +
                         'earliest one); default: %(default)d).')
parser.add_argument('--step',
                    metavar='INT',
                    type=int,
                    default=1,
                    help='The step size in marching the snapshots (default: ' +
                         '%(default)d).')
# moved to environment DEFAULT-values for cache-processing
# parser.add_argument('--linking-length', '-l',
#                     metavar='FLOAT',
#                     type=float,
#                     default=0.05,
#                     help='The linking length used for the FoF finder in terms ' +
#                          'of the mean particle separation. ' +
#                          'Default: %(default)g.')
# parser.add_argument('--linking-vel', '-lv',
#                     metavar='STR',
#                     default='100 km/s',
#                     help='The linking velocity used for the FoF finder. ' +
#                          'Default: %(default)s')
# parser.add_argument('--lowresthreshold',
#                     metavar='FLOAT',
#                     type=float,
#                     default=1e-2,
#                     help='The maximum allowed mass of low resolution elements ' +
#                          'in a halo, defined by a sphere of R200 around the ' +
#                          'center of mass of the FoF group. (default: ' +
#                          '%(default)g).')
# parser.add_argument('--gal-rad',
#                     metavar='FLOAT',
#                     type=float,
#                     default=0.1,
#                     help='The radius in units of R200 to define the galaxy ' +
#                          '(default: %(default)s).')
parser.add_argument('--tlimit',
                    metavar='STR',
                    default='23 h',
                    help='If this time limit in hours is exceeded, stop the ' +
                         'caching (default: %(default)s).')
parser.add_argument('--overwrite',
                    action='store_true',
                    help='Force to overwrite possibly existing files in the ' +
                         'trace folder.')
parser.add_argument('--verbose', '-v',
                    action='store_true',
                    help='Run in verbose mode.')
# writing to snapshot files not supported
# parser.add_argument('--cut',
#                     metavar='FLOAT',
#                     type=float,
#                     default=None,
#                     help='Also save the cut halos (all within <cut>*R200, not ' +
#                          'the FoF group) as Gadget files in the trace folder.')
# parser.add_argument('--length',
#                     default='ckpc/h_0',
#                     help='Define the Gadget length units (default: %(default)s).')
# parser.add_argument('--mass',
#                     default='1e10 Msol h_0**-1',
#                     help='Define the Gadget mass units (default: %(default)s).')
# parser.add_argument('--velocity',
#                     default='km/s',
#                     help='Define the Gadget velocity units (default: %(default)s).')


# def my_center(halo, galaxy, snap, old_center, args):
#     '''The definition of the center of the galaxy/halo.'''
#     if galaxy is not None:
#         center = galaxy.com
#     elif halo is not None:
#         # take all matter within args.gal_rad * R200 like in the definition of the
#         # non-FoF galaxy:
#         R200, M200 = pg.analysis.virial_info(snap, center=old_center, odens=200)
#         inner_halo = snap[pg.BallMask(center=old_center, R=args.gal_rad*R200,
#                                          sph_overlap=False)]
#         center = pg.analysis.center_of_mass(inner_halo)
#         del inner_halo
#     else:
#         center = old_center
#     return center

def find_last_snapshot(args):
    '''Find the highest (consecutive) snapshot number.'''
    i_max = int(1e4)
    for num in range(args.end, i_max):
        nums = (num,) * args.name_pattern.count('%')
        snap_fname = args.name_pattern % nums
        if not os.path.exists(snap_fname):
            if num == args.end:
                snap_fname = os.path.basename(snap_fname)
                print('ERROR: first snapshot "%s"' % snap_fname,
                                     'does not exist!', file=sys.stderr)
                sys.exit(1)
            return num-1

    print('ERROR: did not find last snapshot number in',
                         '[%d,%d] with pattern "%s"!' % (args.end, i_max-1,
                                 os.path.basename(args.name_pattern)), file=sys.stderr)
    sys.exit(1)

def find_first_snapshot(args):
    '''Find the first snapshot number.'''
    i_max = int(1e4)
    for num in range(0, i_max):
        nums = (num,) * args.name_pattern.count('%')
        snap_fname = args.name_pattern % nums
        if not os.path.exists(snap_fname):
            if num == i_max:
                snap_fname = os.path.basename(snap_fname)
                print('ERROR: first snapshot "%s"' % snap_fname,
                                     'does not exist!', file=sys.stderr)
                sys.exit(1)
        else:
            return num

    print('ERROR: did not find first snapshot number in',
                         '[%d,%d] with pattern "%s"!' % (args.end, i_max,
                                 os.path.basename(args.name_pattern)), file=sys.stderr)
    sys.exit(1)


def load_command(script_file):
    command_str = ''
    if script_file == '':
        return command_str
    filename = script_file
    if len(script_file) > 3 and script_file[:-3].lower() != '.py':
        filename = filename + '.py'
    try:
        with open(filename, 'r') as myfile:
            command_str = ''
            ignore = True
            intent = ''
            for line in myfile:
                if len(line) >= 3 and line[:3] == "def": ignore = False
                if line.find('__name__') >= 0 and line.find('__main__'):
                    ignore = False
                elif len(line) >= 15 and line[:15] == "# gcache3-init:":
                    ignore = True
                elif len(line) >= 18 and line[:18] == "# gcache3-prepare:":
                    command_str += "if snap_exec == 'prepare':\n"
                    line = line.lstrip(' ')
                    intent = '    '
                    ignore = False
                elif len(line) >= 18 and line[:18] == "# gcache3-process:":
                    command_str += "if snap_exec == 'process':\n"
                    line = line.lstrip(' ')
                    intent = '    '
                    ignore = False
                elif len(line) >= 16 and line[:16] == "# gcache3-close:":
                    command_str += "if snap_exec == 'close':\n"
                    line = line.lstrip(' ')
                    intent = '    '
                    ignore = False
                elif len(line) >= 14 and line[:14] == "# gcache3-end:":
                    ignore = True
                elif len(line) >= 1 and line[:1] == "#":
                    line = ''
                if not ignore and line != "":
                    command_str += intent + line
            myfile.close()
    except Exception as e:
        pass

    return command_str

def plot_snapshot(sf, snapshot, title=None):
    figx = pg.environment.PLOT_figx
    figy = pg.environment.PLOT_figy
    fs = pg.environment.PLOT_fontsize

    prefix = sf.profile
    print("*********************************************")
    print("plotting snapshot ", snapshot.parts)
    print("prefix, title: ", prefix, ", ", title)
    fig, ax = plt.subplots(figsize=(figx, figy))
    if not title is None:
        ax.set_title(title + ' ('+ prefix + ')')

    pg.plotting.image(snapshot.stars, ax=ax)
    fig.legend(fontsize=fs)
    #if CONFIG.figsave:
    #    plt.savefig('fig-' + prefix + '-snapshot-' + str(title))
    #if CONFIG.figshow:
    #    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    print("plotting complete")
    print("*********************************************")
    return buf

# def write_star_form(snap, virinfo, last_snap, last_R200, args):
#     '''Store the star formation information.'''
#     if last_snap is not None:
#         if args.verbose:
#             print('gather newly formed stars...')
#             sys.stdout.flush()
#
#         stars = last_snap.stars
#         if len(stars) == 0:
#             return
#
#         assert stars['form_time'].units == 'a_form'
#         new_stars = stars[(snap.scale_factor < stars['form_time']) &
#                           (stars['form_time'] <= last_snap.scale_factor)]
#
#         if args.verbose:
#             print('  %d new star particles born' % len(new_stars))
#
#         last_R200_f = float(last_R200)
#         this_R200_f = float(virinfo['R200'])
#         last_a = float(last_snap.scale_factor)
#         this_a = float(snap.scale_factor)
#         IDs = new_stars['ID'].view(np.ndarray)
#         r = new_stars['r'].view(np.ndarray)
#         a = new_stars['form_time'].view(np.ndarray)
#         x = (a - last_a) / (this_a - last_a)
#         R200_a = x * this_R200_f + (1.0-x) * last_R200_f
#         del x
#         r_in_R200 = r / R200_a
#         r_in_kpc = \
#                 new_stars['r'].in_units_of('kpc',subs=last_snap).view(np.ndarray)
#         Z = new_stars['metallicity'].view(np.ndarray)
#         star_form_filename = args.destination + '/star_form.ascii'
#         if args.verbose:
#             print('  write their properties to "%s"...' % star_form_filename)
#             sys.stdout.flush()
#         with open(star_form_filename, 'a') as f:
#             for i in range(len(new_stars)):
#                 print('%-12d \t %-11.9f \t %-18.5g \t %-10.5g \t %-11.9f' % (
#                             IDs[i], a[i], r_in_kpc[i], r_in_R200[i], Z[i]), file=f)

# def write_halo_subsnap(args, snap, cut_R):
#     '''Write the halo as a new snapshot.'''
#     if cut_R:
#         cut_filename = args.destination + ('/snap_cut_%03d.gdt' % snap_num)
#         if args.verbose:
#             print('write cutted region (< %s) to "%s"...' % (cut_R, cut_filename))
#             sys.stdout.flush()
#         if os.path.exists(cut_filename) and not args.overwrite:
#             print('ERROR: file "%s" already ' % cut_filename + \
#                                  'exists (consider "--overwrite")!', file=sys.stderr)
#             sys.exit(1)
#         pg.gadget.write(
#                 snap[pg.BallMask(cut_R, sph_overlap=False)],
#                 cut_filename,
#                 blocks=None,    # -> all blocks
#                 gformat=2, endianness='native',
#                 infoblock=True,
#                 double_prec=False,
#                 gad_units={'LENGTH':args.length,
#                            'MASS':args.mass,
#                            'VELOCITY':args.velocity},
#                 overwrite=args.overwrite,
#         )
#     else:
#         if args.verbose: print('no halo to write')

# # linking length to physical length
# def linking_length_phys(s, Omega, ll):
#     m = np.mean(s['mass'])                  # mean particle mass
#     rho = Omega * s.cosmology.rho_crit(z=0) # mean density
#     # mean particle separation:
#     # (factor 2**(1/6) is assuming a densest sphere packing)
#     d = 2**(1/6.) * (m / rho)**Fraction(1,3)
#     return ll * d.in_units_of('ckpc', subs={'a':1.0,'h_0':s.cosmology.h_0})


if __name__ == '__main__' or __name__ == 'pygad.cmd.gCache': # imported by command line script
    args = parser.parse_args()

    import sys
    if args.verbose:
        print('starting up...')
        sys.stdout.flush()

    # Suspend importing to here, after parsing the arguments to be quick,
    # if only the help has to be displayed or there already occurs and
    # error in parsing the arguments.
    import time
    t_start = time.time()
    import os
    import io
    import gc
    from fractions import Fraction
    import numpy as np
    import pickle as pickle
    import pygad as pg                  # to enable easy testing of command-scripts use absolute

    import matplotlib.pyplot as plt

    start_time = time.time()
    if args.verbose:
        pg.environment.verbose = pg.environment.VERBOSE_TACITURN
    else:
        pg.environment.verbose = pg.environment.VERBOSE_QUIET

    # test arguments
    if args.name_pattern.count('%') == 0:
        if args.verbose:
            print('ERROR: the pattern of the snapshot filename',
                                 'does not contain any string formatter. It',
                                 'will be the same name for all snapshots!', file=sys.stderr)
        sys.exit(1)
    # prepare arguments
    if args.destination is None:
        args.destination = os.path.dirname(args.name_pattern) + '/' + args.profile

    # prepare trace folder
    if not os.path.exists(args.destination):
        if args.verbose: print('create profile folder: "%s"' % args.destination)
        os.makedirs(args.destination)
    if os.listdir(args.destination):
        print('WARNING: profile is not empty!', file=sys.stderr)

    RESTART_FILENAME = args.destination + '/' + RESTART_FILENAME
    if args.overwrite and os.path.exists(RESTART_FILENAME):
        os.remove(RESTART_FILENAME)
    if os.path.exists(RESTART_FILENAME):
        if args.verbose:
            print('reading restart file "%s"...' % RESTART_FILENAME)
            print('WARNING: ignoring --start=%s!' % args.start)
            sys.stdout.flush()
        with open(RESTART_FILENAME, 'rb') as f:
            old_args, snap_num = pickle.load(f)
        # check the compability of the arguments
        new_args = vars(args).copy()
        for not_cmp in ['start', 'step', 'command','withplot', 'end', 'tlimit', 'overwrite', 'verbose']:
            old_args.pop(not_cmp)
            new_args.pop(not_cmp)
        if old_args != new_args:
            print('ERROR: arguments are not compatible ' + \
                                 'with the previous run!', file=sys.stderr)
            print('Not matching these old arguments:', file=sys.stderr)
            for arg, value in old_args.items():
                if arg in new_args and new_args[arg] != value:
                    print('  %-20s %s' % (arg+':', value), file=sys.stderr)
            sys.exit(1)
        del old_args, new_args
        args.start = snap_num-args.step
        if args.verbose:
            print('done reading restart file')
        if args.end is None:
            args.end = find_first_snapshot(args)
        if snap_num == args.end:
            print('WARNING: no snapshot to process last =', snap_num, ' end =', args.end, file=sys.stderr)
            exit(1)

        print('restart: snapshot ', args.start, args.end)

    else:
        if args.end is None:
            args.end = find_first_snapshot(args)

        if args.start is None:
            args.start = find_last_snapshot(args)

        # prepare starformation file
        # star_form_filename = args.destination + '/star_form.ascii'
        # if args.verbose:
        #     print('====================================================')
        #     print('prepare star formation file "%s"...' % star_form_filename)
        #     sys.stdout.flush()
        # if os.path.exists(star_form_filename) and not args.overwrite:
        #     print('ERROR: file "%s" already ' % star_form_filename + \
        #                          'exists (consider "--overwrite")!', file=sys.stderr)
        #     sys.exit(1)
        # with open( star_form_filename, 'w' ) as f:
        #     print('%-12s \t %-11s \t %-18s \t %-10s \t %-11s' % (
        #                 'ID', 'a_form', 'r [kpc]', 'r/R200(a)', 'Z'), file=f)
        #
        # snap_num = args.start
        # print('prepare: snapshot ', snap_num, args.end)
        #
        # if args.verbose:
        #     print('found halo & galaxy of interest in %f sec' % (
        #             time.time() - start_time))


    # do tracing / loop over snapshot (in reverse order)
    if args.verbose: start_time = time.time()

    print('*** process snapshots ', args.start, ' ... ', args.end)
    print('*** prepare')
    snap_exec = 'prepare'
    snap_num = args.start
    snap_start = args.start
    snap_end = args.end

    command_str = load_command(args.command)
    if command_str != '':
        exec(command_str, globals(), locals())

    for snap_num in range(snap_start, snap_end-1, -args.step):
        nums = (snap_num,) * args.name_pattern.count('%')
        snap_fname = args.name_pattern % snap_num
        if args.verbose:
            print('====================================================')
            print('process snapshot #%03d' % snap_num)
            print('process snapshot ', snap_fname)
            sys.stdout.flush()

        dfSnap = pg.SnapshotCache(snap_fname, profile=args.profile)
        print('*** ', snap_num, ' process')
        dfSnap.load_snapshot()

        gx=dfSnap.galaxy
        if args.withplot:
            if not 'galaxy-all' in dfSnap.gx_properties:
                if args.verbose:
                    print("plot particles in galaxy ", gx.parts)
                buf = plot_snapshot(dfSnap, dfSnap.galaxy, 'Galaxy star particles')
                dfSnap.gx_properties.append('galaxy-all', buf)
            else:
                if args.verbose:
                    print("plot exists already in cache")

        snap_exec = 'process'
        if command_str != '':
            exec(command_str, globals(), locals())

        print('*** ', snap_num, ' writing cache')
        dfSnap.write_chache()
        print('*** ', snap_num, ' finished')

        if args.verbose:
            print('writing restart file "%s"...' % RESTART_FILENAME)
            sys.stdout.flush()
        with open(RESTART_FILENAME, 'wb') as f:
            pickle.dump((vars(args),snap_num), f)

        if args.verbose:
            print('all done with snapshot #%03d in ' % snap_num + \
                  '%f sec' % (time.time() - start_time))
            start_time = time.time()
            sys.stdout.flush()

        if time.time()-t_start > pg.UnitScalar(args.tlimit, 's'):
            if args.verbose:
                print()
                print('time limit (%s) exceeded -- stop!' % args.tlimit)
            break

    print('*** close')
    snap_exec = 'close'
    if command_str != '':
        exec(command_str, globals(), locals())

    if args.verbose:
        print()
        print('finished.')

else:
    print('unrecognized module name ', __name__ )
