# these packages are loaded by gcache3 by default
import os
import io
import sys
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import pygad as pg

#############################################################################
# additional packages should be loaded in prepare section                   #
# here your code of functions ONLY starts                                   #
#############################################################################

def prepare_snap(snap_cache, cmd_par, cmd_par1):
    # snapshot is already translated and orientated by SnapshotCache
    # ##############################################################
    # snap_num_freq = args.snaps.count('%')
    # snapfile = args.snaps % ((snap_num,) * snap_num_freq)
    # snap = pygad.Snap(snapfile,
    #                   physical=False,
    #                   gad_units={
    #                     'LENGTH':   args.length,
    #                     'VELOCITY': args.velocity,
    #                     'MASS':     args.mass,
    #                   })
    # snap_num_freq = args.infos.count('%')
    # infofile = args.infos % ((snap_num,) * snap_num_freq)
    # info = pygad.tools.read_info_file(infofile)
    #
    # if args.verbose:
    #     print('  ->', snap)
    #     print('  scale param.:', snap.scale_factor)
    #     print('  cosmic time: ', snap.cosmic_time())
    #
    # Ired = info['I_red(gal)']
    # if Ired is None or info['center'] is None:
    #     if args.verbose: print('no central galaxy yet; continue...')
    #     return snap, None
    # Ired = Ired.reshape((3,3))
    #
    # if args.verbose:
    #     print('center (also in velocity) and orientate the snapshot...')
    # if args.verbose:
    #     print('   center at:', info['center'])
    # pygad.Translation(-info['center']).apply(snap)
    # rind = np.argsort(snap['r'])
    # snap['vel'] -= pygad.analysis.mass_weighted_mean(
    #                 snap[snap['r'] < snap['r'][rind[1001]]], 'vel')
    # if args.verbose:
    #     print('   orientation is at reduced inertia tensor')
    # pygad.analysis.orientate_at(snap, 'red I', qty=Ired, total=True)

    snap = snap_cache.snapshot
    snap.to_physical_units()       # TODO not yet clear

    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('restrict to region...')
    if cmd_par == 'disc':
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("  ... the disc (jz/jc>%.2f," % cmd_par1, end=' ')
            print("rmax=0.15*R200, zmax='5 kpc')")
            print("  ... where jc is the angular momentum of a particle with", end=' ')
            print("the same energy, but on a perfectly circular orbit; only", end=' ')
            print("co-rotating particles are counted")
        R200, M200 = pg.analysis.virial_info(snap)
        region = snap[ pg.DiscMask(jzjc_min=cmd_par1,
                                      rmax=0.15*R200, zmax='5 kpc') ]
    elif cmd_par == 'ball':
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("  ... all within %g%% of R200" % (100.*cmd_par1))
        R200, M200 = pg.analysis.virial_info(snap)
        region = snap[ pg.BallMask(cmd_par1*R200) ]
    elif cmd_par == 'disc-Uebler':
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("  ... the disc (|jz/jc| in [0.8,1.2], rmax=0.15*R200, zmax='3 kpc')")
            print("  ... where jc = r * sqrt(G*M(<r)/r)")
        R200, M200 = pg.analysis.virial_info(snap)
        r_sort_arg = np.argsort(snap['r'])
        M_enc = np.cumsum( snap['mass'][r_sort_arg] )\
                [pg.utils.perm_inv(r_sort_arg)]
        jc = snap['mass'] * snap['r'] * np.sqrt( pg.physics.G * M_enc / snap['r'] )
        del r_sort_arg, M_enc
        jz_jc = (snap['angmom'][:,2]/jc).in_units_of(1,subs=snap)
        rmax = (0.15*R200).in_units_of(snap['r'].units, subs=snap)
        zmax = pg.UnitArr('3 kpc').in_units_of(snap['z'].units, subs=snap)
        region = snap[ (0.8 < np.abs(jz_jc)) & (np.abs(jz_jc) < 1.2) &
                       (snap['r'] < rmax) & (snap['z'] < zmax) ]
        del jc, jz_jc
    else:
        raise RuntimeError('Unkown region mode "%s"!' % cmd_par)

    return snap, region


def add_state_info( sub, new=False, vel=False, temp=True, info='' ):
    dtypel = [('ID', 'uint64'), ('a', 'float64'), ('mass', 'float64'),
              ('metals', 'float64'), ('angmom','float64')]
    if temp:
        dtypel.append(('temp', 'float64'))
    if vel:
        dtypel.append(('vel','float64'))

    data = np.zeros((len(sub),), dtype=np.dtype(dtypel))
    data['ID'] = sub['ID']
    data['a'] = sub.scale_factor * np.ones(len(sub),float)
    data['mass'] = sub['mass']
    data['metals'] = sub['metals']
    data['angmom'] = sub['angmom'][:,2]
    # data = [sub['ID'],
    #         sub.scale_factor * np.ones(len(sub),float),
    #         sub['mass'],
    #         sub['metals'],
    #         sub['angmom'][:,2]]
    if temp:
        # data.append( sub['temp'] )
        if 'temp' in sub.available_blocks():
            data['temp'] = sub['temp']
    if vel:
        # data.append( np.linalg.norm(sub['vel'],axis=-1) )
        if 'vel' in sub.available_blocks():
            data['vel'] = np.linalg.norm(sub['vel'],axis=-1)

    # data = np.vstack(data)
    # data = np.transpose(data)
    if new:
        for row in data:
            trace_data[int(row[0])] = [('n', list(row)[1:])]
    else:
        for row in data:
            ID = int(row[0])
            #trace_data[ID] = np.concatenate( (trace_data[ID], list(row)[1:]) )
            trace_data[ID].append((info, list(row)[1:]))
            #print("append add_state_info ",ID, len(list(row)[1:]), " ", info)
    return


def add_state_missing( missing_IDs, sub):
    dtypel = [('ID', 'uint64'), ('a', 'float64')]

    data = np.zeros((len(missing_IDs),), dtype=np.dtype(dtypel))
    data['ID'] = np.array(list(missing_IDs), dtype=np.uint64)
    data['a'] = sub.scale_factor * np.ones(len(missing_IDs),float)
    for row in data:
        ID = int(row[0])
        trace_data[ID].append(('v', list(row)[1:]))
    return


def start_trace_new( region, region_IDs ):
    '''Start to trace particles that are new in region.'''
    new_IDs = region_IDs - traced_IDs
    traced_IDs.update(new_IDs)
    new = region.gas[ pg.IDMask(new_IDs) ]
    if len(set(new['ID'])) - len(set(new_IDs)) != 0:
        print("FEHLER start_trace_new Mask ", len(set(new['ID'])), len(set(new_IDs)) )
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('new gas particles to trace:             %-10d' % len(new))
    add_state_info( new, new=True, info="n" )
    return


def stop_trace_stars( stars ):
    '''Stop tracing particles that turned into stars.'''
    new_star_IDs = set(stars['ID']).intersection(traced_IDs)
    traced_IDs.difference_update( new_star_IDs )
    new_stars = snap.stars[ pg.IDMask(new_star_IDs) ]
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('traced gas particles that formed stars: %-10d' % len(new_stars))
    add_state_info( new_stars, temp=False, info="s"  )
    return


def deactivate_reentering( region, region_IDs ):
    reenter_IDs = out_IDs.intersection(region_IDs)
    out_IDs.difference_update( reenter_IDs )
    reenter = region[ pg.IDMask(reenter_IDs) ]
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('gas particles re-entered the region:    %-10d' % len(reenter))
    add_state_info( reenter , info="r")
    return


def activate_leaving( snap, region_IDs ):
    leaving_IDs = (traced_IDs - out_IDs) - region_IDs
    out_IDs.update( leaving_IDs )
    leaving = snap.gas[ pg.IDMask(leaving_IDs) ]
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('gas particles leaving the region:       %-10d' % len(leaving))
    add_state_info( leaving, vel=True, info="l" )
    missing_IDs = leaving_IDs - set(leaving['ID'])
    add_state_missing(missing_IDs, leaving)

    dtypel = [('ID', 'uint64'), ('a1', 'float64'), ('r', 'float64'),
              ('a2', 'float64'), ('pos','float64')]
    data = np.zeros((len(leaving),), dtype=np.dtype(dtypel))
    data['ID'] = leaving['ID']
    data['a1'] = leaving.scale_factor * np.ones(len(leaving),float)
    data['r'] = leaving['r']
    data['a2'] = leaving.scale_factor * np.ones(len(leaving),float)
    data['pos'] = leaving['pos'][:,2]
    # data = [leaving['ID'],
    #         leaving.scale_factor * np.ones(len(leaving),float),
    #         leaving['r'],
    #         leaving.scale_factor * np.ones(len(leaving),float),
    #         leaving['pos'][:,2]]
    # data = np.vstack( data ).T
    for row in data:
        ID = int(row[0])
        #trace_data[ID] = np.concatenate( [trace_data[ID], list(row)[1:]] )
        trace_data[ID].append(('p', list(row)[1:]))
        #print("Append activate_leaving ", ID, len(list(row)[1:]))
    return


def update_activated( snap ):
    active = snap.gas[ pg.IDMask(out_IDs) ]
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('update traced particles outside:        %-10d' % len(out_IDs))

    dtypel = [('ID', 'uint64'), ('a1', 'float64'), ('r', 'float64'),
              ('a2', 'float64'), ('pos', 'float64')]
    current = np.zeros((len(active),), dtype=np.dtype(dtypel))
    current['ID'] = active['ID']
    current['a1'] = active.scale_factor * np.ones(len(active),float)
    current['r'] = active['r']
    current['a2'] = active.scale_factor * np.ones(len(active),float)
    current['pos'] = active['pos'][:,2]
    # current = [active['ID'],
    #            active.scale_factor * np.ones(len(active),float),
    #            active['r'],
    #            active.scale_factor * np.ones(len(active),float),
    #            np.abs(active['pos'][:,2])]
    # current = np.vstack( current ).T
    for row in current:
        data = trace_data[int(row[0])]
        if data[-3] < list(row)[-3]:
            data[-4:-2] = list(row)[-4:-2]
        if data[-1] < list(row)[-1]:
            data[-2:] = list(row)[-2:]
        #print("Concatenate update_activated ", row[0], len(data))
    return


def get_trace_file_name():
    return snap_cache.get_profile_path() + cmd_par2

##############################################################################
# here start the gcache3 sections, invoking your code                        #
##############################################################################
# the order of the gcache3 sections is : init, prepare, process, close, end  #
# sections are identified by line starting with # gcache3-<section name>:    #
# the order must not be modified                                             #
##############################################################################

##############################################################################
# gcache3-init: ignored by gcache3
##############################################################################
pg.environment.verbose=pg.environment.VERBOSE_QUIET
os.environ['SNAPSHOT_HOME'] = '/mnt/hgfs/AstroDaten/CosmoZooms/'
os.environ['SNAPCACHE_HOME'] = '/mnt/hgfs/Astro/CosmoCache/'

snap_fname_rel = "M0094/AGN/snap_m0094_sf_x_2x_060"
snap_cache = pg.SnapshotCache(snap_fname_rel, profile='FullRun')
snap_cache.load_snapshot()

snap_num = 94
snap_start = 94
snap_end = 94
snap_overwrite = True

# parser.add_argument('--region',
#                     choices=['disc', 'ball', 'disc-Uebler'],
#                     default='ball',
#                     help='The definition of the region (default: %(default)s). ' +
#                          '"disc" is everything with jz/jc>0.85, r<0.15*R200, ' +
#                          'and |z|<5 kpc, where jc is the angular momentum of a ' +
#                          'particle on a circular orbit with the same energy, ' +
#                          'only co-rotating particles are counted; "ball" is ' +
#                          'just everything within 0.10*R200; "disc-Uebler" is ' +
#                          'the definition of Uebler+ (2014): 0.8<|jz/j\'c|<1.2, ' +
#                          'r<0.15*R200, and |z|<5 kpc, where ' +
#                          'j\'c := r * sqrt(G*M(<r)/r).')
# parser.add_argument('--region_param',
#                     default=None,
#                     help='A parameter for the region. For "disc" it is the ' +
#                          'critical jz/jc value (default: 0.85); for "ball" it ' +
#                          'is the radius in terms of the virial radius ' +
#                          '(default: 10%%), for "disc-Uebler" it is ignored.')
cmd_par = 'gx'      # region
cmd_par1 = '0.1'     # region_param
cmd_par2 = 'traced_gas.dat'

##############################################################################
# gcache3-prepare: processes before first snapshot
##############################################################################

# Parameter not yet clear, how to migrate
# parser.add_argument('--length',
#                     default='ckpc/h_0',
#                     help='Define the Gadget length units (default: %(default)s).')
# parser.add_argument('--mass',
#                     default='1e10 Msol h_0**-1',
#                     help='Define the Gadget mass units (default: %(default)s).')
# parser.add_argument('--velocity',
#                     default='km/s',
#                     help='Define the Gadget velocity units (default: %(default)s).')

print("command CmdTraceGas - prepare ", snap_num, " von ", snap_start, " bis ", snap_end)
#pg.environment.verbose = pg.environment.VERBOSE_TACITURN
if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    print('starting up...')

#trace_file_name = get_trace_file_name()
#if os.path.exists(trace_file_name) and not snap_overwrite:
#    print('ERROR: destination path exsits ' + \
#       '(%s)! Consider "--overwrite".' % trace_file_name, file=sys.stderr)
#    sys.exit(1)

# trace (gas) particles by ID
trace_data = {}
traced_IDs = set()
out_IDs = set()
snap = None

globals()['trace_data'] = trace_data
globals()['traced_IDs'] = traced_IDs
globals()['out_IDs'] = out_IDs
globals()['snap'] = snap

if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    start_total = time.time()

# processing some arguments
if cmd_par is None:
    cmd_par = 'gx'
if cmd_par == 'disc':
    if cmd_par1 is None: cmd_par1 = 0.85
    cmd_par1 = float(cmd_par1)
elif cmd_par == 'ball':
    if cmd_par1 is None: cmd_par1 = 0.1
    cmd_par1 = float(cmd_par1)
elif cmd_par == 'disc-Uebler':
    pass  # no use of cmd_par1
elif cmd_par == 'gx':
    pass # region of galaxy as set by SnapshotCache
else:
    raise RuntimeError('Unkown region mode "%s"!' % cmd_par)

# Suspend importing to here, after parsing the arguments to be quick,
# if only the help has to be displayed or there already occurs and
# error in parsing the arguments.

if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    print('imported pygad', pg.version)

# not necessary, covered by gCache
# if os.path.exists(args.dest) and not args.overwrite:
#     print('ERROR: destination path exsits ' + \
#           '(%s)! Consider "--overwrite".' % args.dest, file=sys.stderr)
#     sys.exit(1)

##############################################################################
# gcache3-process: processes for each snapshot
##############################################################################
print("command CmdTraceGas - process", snap_num)

start_loop = time.time()
if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    print('====================================================')
    print('read snapshot and trace file #%03d...' % snap_num)
    sys.stdout.flush()
if cmd_par == 'gx':
    snAllIDs = snap_cache.snapshot['ID']
    snIDs, snCnt = np.unique(snAllIDs,return_counts=True)
    snDict = dict([ (ID, Cnt) for ID, Cnt in zip(snIDs, snCnt)])
    snUMask =np.array([(snDict[ID] == 1) for ID in snAllIDs], dtype=bool)
    snap = snap_cache.snapshot[snUMask]
    if snap_cache.galaxy is None:
        region = None
    else:
        gxAllIDs = snap_cache.galaxy['ID']
        gxIDs, gxCnt = np.unique(gxAllIDs,return_counts=True)
        gxDict = dict([ (ID, Cnt) for ID, Cnt in zip(gxIDs, gxCnt)])
        gxUMask = np.array([(gxDict[ID] == 1 and snDict[ID] == 1) for ID in gxAllIDs], dtype=bool)
        region = snap_cache.galaxy[gxUMask]
else:
    snap, region = prepare_snap(snap_cache, cmd_par, cmd_par1)

if region is not None:  # central galaxy
    region_IDs = set(region.gas['ID'])

    if len(set(region_IDs) - set(snap['ID'])) > 0:
        print("FEHLER Region_IDs - ", set(region_IDs) - set(snap['ID']))

    # continue as long as there is not / has not been a region with
    # at least 100 particles
    if not trace_data and len(region.baryons) < 100:
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('no region yet (only considered defined when it has more')
            print('than 100 baryonic particles)')
            print('continue')
    else:
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print(r'region has a stellar mass of %.2g %s (%d particles)' % (
                region.stars['mass'].sum(), region.stars['mass'].units,
                len(region.stars)))
            print(r'   ... and a gas mass of %.2g %s (%d particles)' % (
                region.gas['mass'].sum(), region.gas['mass'].units, len(region.gas)))
            sys.stdout.flush()
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('currently tracing %d particles...' % len(traced_IDs))

        start_trace_new(region, region_IDs)
        stop_trace_stars(snap.stars)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('... now tracing %d particles' % len(traced_IDs))

        deactivate_reentering(region, region_IDs)
        activate_leaving(snap, region_IDs)
        #update_activated(snap)

if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    print('done in %.3gs' % (time.time() - start_loop))

snap_cache.gx_properties.append('gashistoryfile', cmd_par2)

print("command CmdTraceGas - process ready", snap_num, " len=", len(trace_data))

##############################################################################
# gcache3-close: processed after last snapshot
##############################################################################
print("command CmdTraceGas - close")
if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
    print('====================================================')
    print('store results...')
    sys.stdout.flush()

gas_trace_filename = get_trace_file_name()
with open(gas_trace_filename, 'wb') as f:
    pickle.dump(trace_data, f, pickle.HIGHEST_PROTOCOL)


##############################################################################
# gcache3-end: ignored by gcache3
##############################################################################


