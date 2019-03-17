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

def write_header(col_list, fcat):
    print("CmdCatalog - write headline")
    line = "filename, Snapshot"
    for col in col_list:
        line += "," + col
    line += "\n"
    fcat.write(line)

def write_line(snapfile, col_list, fcat):
    line = snap_fname + "," + str(snap_num)
    for col in col_list:
        if col in snapfile.gx_properties:
            value = snapfile.gx_properties[col]
            if isinstance(value, pg.UnitArr):
                value = float(value.view(np.ndarray))
        else:
            value = " "
        line += "," + str(value)
    line += "\n"
    fcat.write(line)
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
snap_cache = pg.SnapshotCache("/mnt/hgfs/Astro/Magneticum/M0094/AGN/snap_m0094_sf_x_2x_094",
                          profile='FastRun')
snap_cache.load_snapshot()

snap_fname = "/mnt/hgfs/Astro/Magneticum/M0094/AGN/snap_m0094_sf_x_2x_094"
snap_num = 94
snap_start = 94
snap_end = 94
snap_overwrite = True
cmd_destination = os.getenv('SNAPCACHE_HOME', None)
cmd_par = "catalog.txt"
cmd_par1 = 'a,redshift,R200,M200,M_stars,R_half_M,R_half_M(faceon),R_eff(faceon),D/T(stars),jzjc_baryons,gamma-halo'

##############################################################################
# gcache3-prepare: processes before first snapshot
##############################################################################
#print("command CmdCatalog - prepare ", snap_num, " von ", snap_start, " bis ", snap_end)
pass

##############################################################################
# gcache3-process: processes for each snapshot
##############################################################################

col_names = split_cols(cmd_par1)
if cmd_par == '':
    fname = "catalog.txt"
else:
    fname = cmd_par

if cmd_destination is None:
    fname = snap_cache.get_profile_path() + fname
else:
    fname = cmd_destination + fname

if os.path.exists(fname):
    neu = False
else:
    neu = True

if neu:
    fcat = open(fname,"w")
    write_header(col_names, fcat)
else:
    fcat = open(fname, "a+")

#print("command CmdCatalog - write snapshot ", snap_num, " to catalog ", fname)
write_line(snap_cache, col_names, fcat)
fcat.close()

##############################################################################
# gcache3-close: processed after last snapshot
##############################################################################

#print("command CmdCatalog - finished ")
pass

##############################################################################
# gcache3-end: ignored by gcache3
##############################################################################


