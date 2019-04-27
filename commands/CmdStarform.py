# these packages are loaded by gcache3 by default
import os
import io
import sys
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import pygad as pg


# additional packages should be loaded in prepare section
# here your code of functions ONLY starts
def split_cols(col_names):

    col_list = col_names.split(',')
    return col_list

def write_header(fstarform):
    print('%-12s \t %-11s \t %-18s \t %-10s \t %-11s' % (
        'ID', 'a_form', 'r [kpc]', 'r/R200(a)', 'Z'), file=fstarform)

def write_star_form(snapfile, last_snap, last_R200, fstarform):
    '''Store the star formation information.'''
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('gather newly formed stars...')
        sys.stdout.flush()

    stars = last_snap.snapshot.stars
    if len(stars) == 0:
        return

    assert stars['form_time'].units == 'a_form'
    new_stars = stars[(snapfile.snapshot.scale_factor < stars['form_time']) &
                      (stars['form_time'] <= last_snap.snapshot.scale_factor)]

    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('  %d new star particles born' % len(new_stars))

    last_R200_f = float(last_R200)
    this_R200_f = float(snap_cache.gx_properties['R200'])
    last_a = float(last_snap.snapshot.scale_factor)
    this_a = float(snap_cache.snapshot.scale_factor)
    IDs = new_stars['ID'].view(np.ndarray)
    r = new_stars['r'].view(np.ndarray)
    a = new_stars['form_time'].view(np.ndarray)
    x = (a - last_a) / (this_a - last_a)
    R200_a = x * this_R200_f + (1.0-x) * last_R200_f
    del x
    r_in_R200 = r / R200_a
    r_in_kpc = new_stars['r'].in_units_of('kpc',subs=last_snap.snapshot).view(np.ndarray)
    Z = new_stars['metallicity'].view(np.ndarray)
    if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
        print('  write their properties to "%s"...' % star_form_filename)
        sys.stdout.flush()
    print("number of new stars ", len(new_stars))
    for i in range(len(new_stars)):
        print('%-12d \t %-11.9f \t %-18.5g \t %-10.5g \t %-11.9f' % (
                    IDs[i], a[i], r_in_kpc[i], r_in_R200[i], Z[i]), file=fstarform)
    return



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

snap_fname_rel = "M0094/AGN/snap_m0094_sf_x_2x_094"
snap_cache = pg.SnapshotCache(snap_fname_rel, profile='FastRun')
snap_cache.load_snapshot()

snap_num = 94
snap_start = 94
snap_end = 94
snap_overwrite = True
cmd_par = ""
last_snap = None

##############################################################################
# gcache3-prepare: processes before first snapshot
##############################################################################
print("command CmdStarform - prepare ", snap_num, " von ", snap_start, " bis ", snap_end)

first_snapshot = True

last_R200 = 0.0

##############################################################################
# gcache3-process: processes for each snapshot
##############################################################################

if cmd_par == '':
    fname = "star_form.ascii"
else:
    fname = cmd_par

star_form_filename = snap_cache.get_profile_path() + fname

if snap_overwrite and snap_num ==  snap_start:
    fstarform = open(star_form_filename, "w")
    write_header(fstarform)
else:
    fstarform = open(star_form_filename, "a+")

print("command CmdStarform - write star forming file ", star_form_filename)
if last_snap is None:
    last_R200 = snap_cache.gx_properties['R200']
else:
    write_star_form(snap_cache, last_snap, last_R200, fstarform)
    last_R200 = snap_cache.gx_properties['R200']

snap_cache.gx_properties.append('starformingfile', fname)

first_snapshot = False

##############################################################################
# gcache3-close: processed after last snapshot
##############################################################################
fstarform.close()
print("command CmdStarform - close last_R200=", last_R200)

##############################################################################
# gcache3-end: ignored by gcache3
##############################################################################


