import os
import io
from PIL import Image
import sys
import re
import time
import numpy as np
import pygad as pg            # use absolute package loading
from .. import environment

__all__ = ["SnapshotCache", "SnapshotProperty"]

class SnapshotCache:
    def __init__(self, snapfile_name, profile='cache'):
        # type: (str, str) -> SnapshotCache
        pg.environment.verbose = environment.verbose
        if snapfile_name[0] != '/':
            snap_home = os.getenv('SNAPSHOT_HOME', '')
            if snap_home == '':
                # treat local path same as absolute path
                data_file = os.path.abspath(snapfile_name)
                cache_home = ''
            else:
                if snap_home[-1] != '/': snap_home = snap_home + '/'
                data_file = snap_home + snapfile_name
                cache_home = os.getenv('SNAPCACHE_HOME', '')
                if cache_home == '':
                    cache_home = snap_home
        else:
            data_file = snapfile_name
            snap_home = ''
            cache_home = ''

        self.__snap_home = snap_home
        self.__cache_home = cache_home
        self.__snap_name = snapfile_name
        self.__data_file = data_file
        self.__profile = ''
        self.__profile_properties = SnapshotProperty()
        self._load_profile(profile)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("************************************************************************")
            print("using snapfile ",self.__data_file)
            print("************************************************************************")

        self.__results = dict()
        self.__snapshot = None
        self.__dm_halo = None
        self.__halo_properties = SnapshotProperty()
        self.__gx_properties = SnapshotProperty()
        self.__galaxy = None
        self.__galaxy_all = None
        self.__family = None
        self.__center = None
        self.__loaded = False
        # #########################################
        # pygad block-names
        # #########################################
        self.__B_POS  = None        # 'pos'       # POS = (3, 'float', 'all')
        self.__B_VEL  = None        # 'vel'       # VEL = (3, 'float', 'all')
        self.__B_ID   = None        # 'ID'        # ID = (1, 'uint', 'all')
        self.__B_MASS = None        # 'mass'      # MASS = (1, 'float', None)
        self.__B_U    = None        # 'u'         # U = (1, 'float', 'gas')
        self.__B_RHO  = None        # 'rho'       # RHO = (1, 'float', 'gas')
        self.__B_HSML = None        # 'hsml'      # HSML = (1, 'float', 'gas')
        self.__B_AGE  = None        # 'age'       # AGE = (1, 'float', None)

        # ##########################################
        # derived block-names (always cached age, temp, mag*, Ekin, angmom, jcirc, LX
        # ##########################################
        self.__D_P    = None        # 'P'          pressure
        self.__D_temp = None        # 'temp'
        self.__D_Ekin = None        # 'Ekin'
        self.__D_Epot = None        # 'Epot'
        self.__D_E    = None        # 'E'
        self.__D_r    = None        # 'r'         # spherical radius
        self.__D_rcyl = None        # 'rcyl'      # dist(pos[:, :2]); cylindrical radius
        self.__D_vrad = None        # 'vrad'      # inner1d(pos, vel) / r;  radial velocities
        self.__D_momentum = None        # 'momentum' # (mass * vel.T).T; momentum
        self.__D_angmom   = None        # 'angmom'   # UnitQty(cross(pos, momentum), pos.units * momentum.units); angular momentum
        # the angular momentum a particle on a circular orbit with the same energy would have
        self.__D_jcirc = None        # 'jcirc'    # r * sqrt(2.0 * mass * Ekin)
        # the parameter jz/jc, where jz is the z-component of the angular momentum and jc=jcirc:
        self.__D_jzjc = None        # 'jzjc'      # angmom[:, 2] / jcirc
        self.__D_vcirc= None        # 'vcirc'      # sqrt(sum(vel ** 2, axis=-1) - vrad ** 2); circular part of the velocities
        # derived from tracing files:
        self.__D_insitu = None        # 'insitu'  # rR200form < 0.1

        self.__L_lum   = None        # 'lum'
        self.__L_lum_u = None        # 'lum_u'
        self.__L_lum_b = None        # 'lum_b'
        self.__L_lum_v = None        # 'lum_v'
        self.__L_lum_r = None        # 'lum_r'
        self.__L_lum_k = None        # 'lum_k'
        self.__L_LX    = None        # 'LX'
        self.__L_mag   = None        # 'mag'
        self.__L_mag_u = None        # 'mag_u'
        self.__L_mag_b = None        # 'mag_b'
        self.__L_mag_v = None        # 'mag_v'
        self.__L_mag_r = None        # 'mag_r'
        self.__L_mag_k = None        # 'mag_k'

    @property
    def B_POS(self):
        # type: () -> pg.SimArr
        return self.galaxy['pos']

    @property
    def B_VEL(self):
        # type: () -> pg.SimArr
        return self.__galaxy['vel']

    @property
    def B_ID(self):
        # type: () -> pg.SimArr
        return self.__galaxy['ID']

    @property
    def B_MASS(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mass']

    @property
    def B_U(self):
        # type: () -> pg.SimArr
        return self.__galaxy['u']

    @property
    def B_RHO(self):
        # type: () -> pg.SimArr
        return self.__galaxy['rho']

    @property
    def B_HSML(self):
        # type: () -> pg.SimArr
        return self.__galaxy['hsml']

    @property
    def B_AGE(self):
        # type: () -> pg.SimArr
        return self.__galaxy['age']

    @property
    def D_P(self):
        # type: () -> pg.SimArr
        return self.__galaxy['P']

    @property
    def D_temp(self):
        # type: () -> pg.SimArr
        return self.__galaxy['temp']

    @property
    def D_Ekin(self):
        # type: () -> pg.SimArr
        return self.__galaxy['Ekin']

    @property
    def D_Epot(self):
        # type: () -> pg.SimArr
        return self.__galaxy['Epot']

    @property
    def D_E(self):
        # type: () -> pg.SimArr
        return self.__galaxy['E']

    @property
    def D_r(self):
        # type: () -> pg.SimArr
        return self.__galaxy['r']

    @property
    def D_rcyl(self):
        # type: () -> pg.SimArr
        return self.__galaxy['rcyl']

    @property
    def D_vrad(self):
        # type: () -> pg.SimArr
        return self.__galaxy['vrad']

    @property
    def D_momentum(self):
        # type: () -> pg.SimArr
        return self.__galaxy['momentum']

    @property
    def D_angmom(self):
        # type: () -> pg.SimArr
        return self.__galaxy['agmom']

    @property
    def D_jcirc(self):
        # type: () -> pg.SimArr
        return self.__galaxy['jcirc']

    @property
    def D_jzjc(self):
        # type: () -> pg.SimArr
        return self.__galaxy['jzjc']

    @property
    def D_vcirc(self):
        # type: () -> pg.SimArr
        return self.__galaxy['vcirc']

    @property
    def D_insitu(self):
        # type: () -> pg.SimArr
        return self.__galaxy['insitu']

    @property
    def L_lum(self):
        # type: () -> pg.SimArr
        return self.__galaxy['lum']

    @property
    def L_lum_u(self):
        # type: () -> pg.SimArr
        return self.__galaxy['lum_u']

    @property
    def L_lum_b(self):
        # type: () -> pg.SimArr
        return self.galaxy['lum_b']
    @property
    def L_lum_v(self):
        # type: () -> pg.SimArr
        return self.__galaxy['lum_v']

    @property
    def L_lum_r(self):
        # type: () -> pg.SimArr
        return self.__galaxy['lum_r']

    @property
    def L_lum_k(self):
        # type: () -> pg.SimArr
        return self.__galaxy['lum_k']

    @property
    def D_LX(self):
        # type: () -> pg.SimArr
        return self.__galaxy['LX']

    @property
    def L_mag(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag']

    @property
    def L_mag_u(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag_u']

    @property
    def L_mag_b(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag_b']

    @property
    def L_mag_v(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag_v']

    @property
    def L_mag_r(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag_r']

    @property
    def L_mag_k(self):
        # type: () -> pg.SimArr
        return self.__galaxy['mag_k']

    # ##########################################
    # Data processing properties and methods
    # ##########################################

    @property
    def name(self):
        return self.__data_file

    @property
    def halo(self):
        if self.__dm_halo is None:
            Rvir_property = self.__profile_properties['Rvir_property']
            if Rvir_property in self.__halo_properties:
                Rvir = self.__halo_properties[Rvir_property]
                mask = pg.BallMask(Rvir, center=[0, 0, 0])
                halo = self.__snapshot[mask]
                self.__dm_halo = halo
        return self.__dm_halo

    @property
    def galaxy(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy

    @property
    def family(self):
        return self.__family

    @family.setter
    def family(self, value):
        if value is None:
            self.__galaxy = self.__galaxy_all
        else:
            gx = self.__galaxy_all
            self.__galaxy = eval('gx.' + str(value),globals(),locals())

        self.__family = value

    @property
    def gas(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.gas

    @property
    def stars(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.stars

    @property
    def dm(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.dm

    @property
    def bh(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.bh

    @property
    def gands(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.gands

    @property
    def sandbh(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.sandbh
    @property

    def baryons(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.baryons

    def highres(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.highres

    def lowres(self):
        # type: () -> pg.SubSnapshot
        return self.__galaxy_all.lowres

    @property
    def halo_properties(self):
        return self.__halo_properties

    @property
    def gx_properties(self):
        return self.__gx_properties

    @property
    def profile_properties(self):
        return self.__profile_properties

    @property
    def profile(self):
        profile = self.__profile
        return profile

    @property
    def snapshot(self):
        return self.__snapshot

    @snapshot.setter
    def snapshot(self, value):
        self.__snapshot = value

    @property
    def results(self):
        return self.__results

    @property
    def Rvir_property(self):
        return self.__profile_properties['Rvir_property']

    @Rvir_property.setter
    def Rvir_property(self, value):
        self.__profile_properties.append('Rvir_property', value)

    @property
    def Rvir(self):
        Rvir_property = self.__profile_properties['Rvir_property']
        Rvir_f  = self.__halo_properties[Rvir_property]
        Rvir_unit = pg.UnitScalar(Rvir_f, self.__snapshot['pos'].units, subs=self.__snapshot)
        return Rvir_unit

    @property
    def gx_radius(self):
        gx_rad = self.__profile_properties['gx_radius']
        return gx_rad

    @property
    def Rgx(self):
        Rgx_unit = self.Rvir * self.gx_radius
        return Rgx_unit

    @property
    def center(self):
        return self.__center

    # #################################
    # load one File into np.arrays
    # #################################

    def __get_halo_filename(self):
        postfix_halo = self.__filename + '-halo.info'
        return self.__destination + "/" + postfix_halo

    def __get_gx_filename(self):
        postfix_gx = self.__filename + '-gx.info'
        return self.__destination + "/" + postfix_gx

    def __get_profile_filename(self):
        postfix_profile = 'profile-' + self.__profile + '.info'
        if self.__global_profile:
            profile_path = pg.gadget.general['Profile_dir'] + '/' + postfix_profile
            profile_path = os.path.expandvars(profile_path)
            return profile_path
        else:
            return postfix_profile

    def get_profile_path(self):
        profile_path = self.__destination + '/'
        return profile_path

    def exists_in_cache(self):
        try:
            self.__halo_properties._read_info_file(self.__get_halo_filename())
            self.__gx_properties._read_info_file(self.__get_gx_filename())
            return True
        except Exception as e:
            return False


    def load_snapshot(self, useCache=True, forceCache=False,
                      loadBinaryProperties=False, loadSF=True, loadGasHistory=False, H_neutral_only=None):
        # type: (bool, bool) -> pg.snapshot
        filename = self.__data_file
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print(("loading snapshot " + filename + "..."))
            print("*********************************************")
        #snapshot = pg.Snap(filename, load_double_prec=True)
        snapshot = pg.Snapshot(filename, load_double_prec=True, H_neutral_only=H_neutral_only)
        self.__snapshot = snapshot

        if 'findgxfast' in self.__profile_properties:
            findgxfast = self.__profile_properties['findgxfast']
        else:
            findgxfast = False

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("boxsize ", snapshot.boxsize)
            print("particles ", snapshot.parts)
            print("headers")
            for h in snapshot.headers()[0]:
                print("......", h, " = ", snapshot.headers()[0][h])
            print("units", snapshot.gadget_units)

        try:
            if not useCache or forceCache:
                raise FileNotFoundError
            # halo properties are found, re-tranlsate and re-orientate the halo, nothing else has to be done
            self.__halo_properties._read_info_file(self.__get_halo_filename())
            self.__gx_properties._read_info_file(self.__get_gx_filename())
            if 'center_in_halo' in self.__gx_properties:
                center_known = self.__gx_properties['center_in_halo']
            else:
                # only dummy profile exists, do not reprocess
                self.__loaded = True
                return self.__snapshot

            self._center_gx(center_known)
            self._gx_orientate()
            if loadBinaryProperties:        # only when reloading, as no binary data in generated automatically
                self.halo_properties.load_binary_properties()
                self.gx_properties.load_binary_properties()
                self.profile_properties.load_binary_properties()
        except Exception as e:
            halo = None
            gx = None
            if not findgxfast:
                halo, gx, halos = self._prepare_zoom(mode='auto', shrink_on='stars', ret_FoF=True)
                if halos is not None:
                    self._create_halo_properties(halos[0])
                else:
                    self._create_halo_properties(None, headeronly=True)
                if gx is not None:
                    self.__galaxy_all = gx
                    self.__galaxy = gx
            else:
                self._create_halo_properties(None)
                self._center_gx(None)
                self._gx_orientate()

            self._create_gx_properties()

            if useCache or forceCache:
                self.__halo_properties._write_info_file(self.__get_halo_filename(), force=forceCache)
                self.__gx_properties._write_info_file(self.__get_gx_filename(), force=forceCache)
                self.__profile_properties._write_info_file(self.__get_profile_filename(), force=forceCache)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("loading complete Rvir=", self.Rvir)
            print("*********************************************")
        self.__loaded = True

        if (loadSF or loadGasHistory):
            if 'redshift' in self.__gx_properties:
                rs = round(float(self.__gx_properties['redshift']), 2)
                a = round(float(self.__gx_properties['a']), 2)
                if True or rs == 0.0:  # load only for z=0
                    if loadSF and "starformingfile" in self.__gx_properties:
                        sfname = self.__gx_properties['starformingfile']
                        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                            print("*********************************************")
                            print("loading star forming information from ", sfname)
                            print("*********************************************")
                        star_form_filename = self.get_profile_path() + sfname
                        self._fill_star_from_info(star_form_filename, a)

                        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                            print("*********************************************")
                            print("reloading derived quanitities")
                            print("*********************************************")
                        self.__snapshot.fill_derived_rules(clear_old=True)

                    if loadGasHistory: # and "gashistoryfile" in self.__gx_properties:
                        sfname = self.__gx_properties['gashistoryfile']
                        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                            print("*********************************************")
                            print("loading gas history information from ", sfname)
                            print("*********************************************")
                        gas_trace_filename = self.get_profile_path() + sfname
                        self._fill_gas_from_traced(self.__snapshot, gas_trace_filename)
        return self.__snapshot


    def read_cache(self, forceCache = False):
        self.__halo_properties._read_info_file(self.__get_halo_filename())
        self.__gx_properties._read_info_file(self.__get_gx_filename())
        return

    def write_cache(self, forceCache = False):
        self.__halo_properties._write_info_file(self.__get_halo_filename(), force=forceCache)
        self.__gx_properties._write_info_file(self.__get_gx_filename(), force=forceCache)
        self.__profile_properties._write_info_file(self.__get_profile_filename(), force=forceCache)
        return

    def clone_profile(self, newprofile, BaseOnly=True, includeGalaxyPlot = True, force=True):
        halo_filename = os.path.split(self.__get_halo_filename())
        halo_dir = os.path.split(halo_filename[0])[0]
        halo_new = halo_dir + '/' + newprofile + '/' + halo_filename[1]

        if os.path.exists(halo_new) and not force:
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("*****************************************************************")
                print("profile ", newprofile, "already exists - clone_profile terminated")
                print("*****************************************************************")
            return

        gx_filename = os.path.split(self.__get_gx_filename())
        gx_dir = os.path.split(gx_filename[0])[0]
        gx_new = gx_dir + '/' + newprofile + '/' + gx_filename[1]
        prop_filename = os.path.split(self.__get_profile_filename())
        prop_new = prop_filename[0] + '/profile-' + newprofile + '.info'

        if force:
            if os.path.exists(halo_new): os.remove(halo_new)
            if os.path.exists(gx_new): os.remove(gx_new)
            if os.path.exists(prop_new): os.remove(prop_new)

        #print("halo:", halo_new)
        #print("gx:", gx_new)
        #print("prop:", prop_new)
        if not os.path.exists(os.path.split(halo_new)[0]):
            os.makedirs(os.path.split(halo_new)[0])
        forceCache = True
        halo_props = ['date-created', 'profile-name', 'dm_only', 'mass', 'Mstars', 'Mgas', 'Mdm',
                      'parts', 'com', 'ssc', 'vel', 'vel_sigma', 'Rmax', 'lowres_part', 'lowres_mass',
                      'Rvir_FoF', 'R200_FoF', 'R200_ssc', 'M200_ssc', 'R200_com', 'M200_com',
                      'R500_FoF', 'R500_ssc', 'M500_ssc', 'R500_com', 'M500_com']

        gx_props = ['date-created', 'profile-name', 'Rvir_property', 'gx_radius', 'findgxfast',
                    'linking_length', 'linking_vel', 'lowres_threshold', 'a', 'cosmic_time', 'redshift',
                    'h', 'center_in_halo', 'center_com', 'center_ssc', 'center_halo_com', 'center_halo_ssc',
                    'R200', 'M200', 'R500', 'M500', 'M_stars', 'L_halo', 'L_baryons', 'M_gas',
                    'Z_stars', 'Z_gas', 'SFR', 'jzjc_baryons', 'jzjc_total', 'I_red',
                    'R_half_M', 'R_half_M_stellar', 'R_half_M(faceon)', 'R_eff(faceon)', 'D/T(stars)']

        #self.__profile_properties._write_info_file(prop_new, force=forceCache)
        new_prop_prop = SnapshotProperty()
        for p in self.__profile_properties:
            new_prop_prop.append(p, self.__profile_properties[p])
        new_prop_prop['profile-name'] = newprofile
        new_prop_prop._write_info_file(prop_new, force=forceCache)

        if BaseOnly:
            new_halo_prop = SnapshotProperty()
            for p in halo_props:
                new_halo_prop.append(p, self.__halo_properties[p])
            new_halo_prop['profile-name'] = newprofile
            new_gx_prop = SnapshotProperty()
            for p in gx_props:
                new_gx_prop.append(p, self.__gx_properties[p])
            new_gx_prop['profile-name'] = newprofile
            if includeGalaxyPlot:
                if 'galaxy-all' in self.__gx_properties:
                    new_gx_prop['galaxy-all'] = self.__gx_properties.get_binary_value('galaxy-all')
            new_halo_prop._write_info_file(halo_new, force=forceCache)
            new_gx_prop._write_info_file(gx_new, force=forceCache)
        else:
            self.__halo_properties['profile-name'] = newprofile
            self.__gx_properties['profile-name'] = newprofile
            self.__halo_properties.load_binary_properties()
            self.gx_properties.load_binary_properties()
            self.__halo_properties._write_info_file(halo_new, force=forceCache)
            self.__gx_properties._write_info_file(gx_new, force=forceCache)

        return


    def execute_command(self, command, par=None, par1=None, par2=None, par3=None, par4=None, par5=None):

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
                print('*** error loading script ', filename)
                print(e)

            return command_str

        import numpy as np
        import matplotlib.pyplot as plt

        # processing as done in gCache, code should be shared instead of copied
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('execute command on snapshot ' + self.__filename)
            print("*** execute command ", command)
            print("*** command parameter ", par)
            print('*** prepare')
        snap_num = 1
        snap_start = 1
        snap_end = 1
        cmd_par = par
        cmd_par1 = par1
        cmd_par2 = par2
        cmd_par3 = par3
        cmd_par4 = par4
        cmd_par5 = par5
        if not self.__loaded:
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('*** load snapshot')
            self.load_snapshot()
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('*** snapshot loaded')
        #dfSnap = self
        globalsParameter = globals().copy()
        globalsParameter['np'] = np
        globalsParameter['plt'] = plt

        localsParameter = locals().copy()
        localsParameter['snap_cache'] = self

        command_str = load_command(command)
        localsParameter['snap_exec'] = 'prepare'
        if command_str != '':
            exec(command_str, globalsParameter, localsParameter)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** process')
        localsParameter['snap_exec'] = 'process'
        if command_str != '':
            exec(command_str, globalsParameter, localsParameter)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** writing cache')
        self.write_cache()
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** finished')

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** close')
        localsParameter['snap_exec'] = 'close'
        if command_str != '':
            exec(command_str, globalsParameter, localsParameter)

        return

    def _load_profile(self, profile):
        self.__global_profile = False
        if profile == '':
            profile_name = 'cache'
        else:
            if len(profile) > 1:
                if profile[:1] == '$':
                    profile = profile[1:]
                    self.__global_profile = True
                else:
                    if not os.path.exists(profile):     # local profile not found
                        self.__global_profile = True    # switch to global mode
            profile_name = profile

        if self.__snap_home == '':
            split_names = os.path.split(self.__data_file)
            if split_names[0] != '':
                self.__destination = split_names[0] + '/' + profile_name
            else:
                self.__destination = profile_name
        else:
            split_names = os.path.split(self.__cache_home + self.__snap_name)
            if split_names[0] != '':
                self.__destination = split_names[0] + '/' + profile_name
            else:
                self.__destination = profile_name

        self.__filename = split_names[1]

        if not os.path.exists(self.__destination):
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('create profle folder: "%s"' % self.__destination)
            os.makedirs(self.__destination)
        if os.listdir(self.__destination):
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('WARNING: profile is not empty!')

        self.__profile = profile_name
        try:
            self.__profile_properties._read_info_file(self.__get_profile_filename())
        except Exception as e:
            # set default profile values
            self.__profile_properties = SnapshotProperty()
            self.__profile_properties.append("date-created", '"' + time.asctime(time.localtime(time.time())) + '"')
            self.__profile_properties.append("profile-name", profile_name)
            self.__profile_properties.append('Rvir_property', environment.DEFAULT_Rvir_property)
            self.__profile_properties.append('gx_radius', environment.DEFAULT_gx_radius)
            self.__profile_properties.append('findgxfast', environment.DEFAULT_findgxfast)
            self.__profile_properties.append('linking_length', environment.DEFAULT_linking_length)
            self.__profile_properties.append('linking_vel', environment.DEFAULT_linking_vel)
            self.__profile_properties.append('lowres_threshold', environment.DEFAULT_lowres_threshold)

        return

    def _correct_virial_info(self, halo_properties):
        def corr_factor(halo_properties, prop, factor):
            if not prop in halo_properties:
                return
            value = halo_properties[prop]
            value = value * factor
            halo_properties[prop] = value
            return

        mass_total = self.__snapshot['mass'].sum()
        mass_dm = self.__snapshot.dm['mass'].sum()
        factor = mass_total/mass_dm
        print("$$$Korrekturfaktor ", mass_total, mass_dm, factor)

        # Rvir_FoF: 299.3049442488739[ckpc h_0 ** -1]
        # R200_FoF: 287.7142082873453[ckpc  h_0 ** -1]
        # R200_ssc: 356.31469894527396[h_0 ** -1 ckpc]
        # M200_ssc: 1.051522e+03[1e+10 h_0 ** -1 Msol]
        # R200_com: 356.29890408445095[h_0 ** -1 ckpc]
        # M200_com: 1.051388e+03[1e+10 h_0 ** -1 Msol]
        # R500_FoF: 211.9896411873922[ckpc h_0 ** -1]
        # R500_ssc: 241.28093820131429[h_0 ** -1 ckpc]
        # M500_ssc: 816.2561863524752[1e+10 h_0 ** -1 Msol]
        # R500_com: 241.27632340717858[h_0 ** -1 ckpc]
        # M500_com: 816.2202291540016[1e+10 h_0 ** -1 Msol]

        corr_factor(halo_properties, 'Rvir_FoF', factor)
        corr_factor(halo_properties, 'R200_FoF', factor)
        corr_factor(halo_properties, 'R200_ssc', factor)
        corr_factor(halo_properties, 'M200_ssc', factor)
        corr_factor(halo_properties, 'R200_com', factor)
        corr_factor(halo_properties, 'M200_com', factor)
        corr_factor(halo_properties, 'R500_FoF', factor)
        corr_factor(halo_properties, 'R500_ssc', factor)
        corr_factor(halo_properties, 'M500_ssc', factor)
        corr_factor(halo_properties, 'R500_com', factor)
        corr_factor(halo_properties, 'M500_com', factor)
        return

    def _create_halo_properties(self, halo, headeronly=False):
        def convert_to_Units(prop):
            if isinstance(prop,tuple):
                value = pg.UnitArr([s for s in prop], units=None)
                return value
            else:
                return prop
        if not headeronly:
            if halo is not None:
                dm_halo = halo
                dm_only = False
            else:   # this part is used, when prepare_zoom is not used
                if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                    print("*********************************************")
                    print(("creating DM-halo ", self.__snapshot.parts))
                    print("*********************************************")

                search_snap = self.__snapshot.dm
                dm_only = True
                # old value 0.05
                if 'lowres_threshold' in self.__profile_properties:
                    threshold = self.__profile_properties['lowres_threshold']
                else:
                    threshold = pg.environment.DEFAULT_lowres_threshold
                # default old parameter linking_length=6 ckpc
                if 'linking_length' in self.__profile_properties:
                    linking_length = self.__profile_properties['linking_length']
                else:
                    linking_length = pg.environment.DEFAULT_linking_length

                if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                    print("search_snap ", search_snap.parts)

                if linking_length is None:
                    linking_length = (1500. * search_snap.cosmology.rho_crit(search_snap.redshift)
                         / np.median(search_snap['mass'])) ** pg.units.Fraction(-1, 3)

                    print("linking length assumed ", linking_length)

                FoF, N_FoF = pg.analysis.find_FoF_groups(search_snap, l=linking_length)
                halos = pg.analysis.generate_FoF_catalogue(search_snap, max_halos=1, FoF=FoF, progressbar=False,
                                                               exlude=lambda h, s: h.lowres_mass / h.mass > threshold)
                dm_halo = halos[0]  # most massive halo

        halo_properties = SnapshotProperty()
        halo_properties.append("date-created", '"' + time.asctime( time.localtime(time.time()) ) + '"')
        halo_properties.append("profile-name", self.__profile)

        if headeronly:
            self.__dm_halo = None
            self.__halo_properties = halo_properties
            return

        # create halo properties
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("DM-halo center ", dm_halo.ssc)
            print("DM-halo properties")

        halo_properties.append("dm_only", dm_only)
        for prop in dm_halo.props:
            v = convert_to_Units(dm_halo.props[prop])
            halo_properties[prop] = v
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("......", prop, " = ", v)

        # if dm_only and dm_halo is not None:
        #     self._correct_virial_info(halo_properties)

        self.__dm_halo = dm_halo
        self.__halo_properties = halo_properties

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("creating complete ")
            print("*********************************************")
        return dm_halo

    def _create_gx_properties(self):

        info = SnapshotProperty()
        info.append("date-created", '"' + time.asctime( time.localtime(time.time()) ) + '"')
        info.append("profile-name", self.__profile)
        if self.__galaxy_all is None:
            self.__gx_properties = info
            return

        # to document calculations copy values to gx-properties
        # theses values should be used for recalculations instead of profile-values
        Rvir_property = self.__profile_properties['Rvir_property']
        gal_radius = self.__profile_properties['gx_radius']
        findgxfast = self.__profile_properties['findgxfast']
        linking_length = self.__profile_properties['linking_length']
        linking_vel = self.__profile_properties['linking_vel']
        lowres_threshold = self.__profile_properties['lowres_threshold']

        info.append('Rvir_property', Rvir_property)
        info.append('gx_radius', gal_radius)
        info.append('findgxfast',findgxfast)
        info.append('linking_length',linking_length)
        info.append('linking_vel',linking_vel)
        info.append('lowres_threshold',lowres_threshold)

        gx = self.__galaxy_all
        halo = self.__snapshot

        R200 = self.halo_properties['R200_com']
        M200 = self.halo_properties['M200_com']

        R500 = self.halo_properties['R500_com']
        M500 = self.halo_properties['M500_com']

        # infos part 1
        info['a'] = self.__snapshot.scale_factor
        info['cosmic_time'] = self.__snapshot.cosmic_time()
        info['redshift'] = self.__snapshot.redshift
        info['h'] = self.__snapshot.cosmology.h(self.__snapshot.redshift)
        #            print('center:             ', center, file=f)
        info['center_in_halo'] = self.__center           # the coord where the gx was found before translation
        info['center_com'] = pg.analysis.center_of_mass(gx) if gx is not None else None
        info['center_ssc'] = pg.analysis.shrinking_sphere(gx, center=[0] * 3,
                                                          R=R200) if gx is not None else None
        info['center_halo_com'] = pg.analysis.center_of_mass(halo) if halo is not None else None
        info['center_halo_ssc'] = pg.analysis.shrinking_sphere(halo, center=[0] * 3,
                                                               R=R200) if halo is not None else None
        info['R200'] = R200
        info['M200'] = M200
        info['R500'] = R500
        info['M500'] = M500

        # infos part 2
        # h = self.__snapshot[pg.BallMask(R200, center=self.__center, sph_overlap=False)]
        # h=halo?
        if halo:
            info['M_stars'] = halo.stars['mass'].sum().in_units_of('Msol', subs=self.__snapshot) \
                                        if len(halo.stars) else None
            info['L_halo'] = halo['angmom'].sum(axis=0)
            info['L_baryons'] = halo.baryons['angmom'].sum(axis=0)
        else:
            info['M_stars'] = pg.UnitArr(0, units='Msol')
            info['L_halo'] = None
            info['L_baryons'] = None
        if gx:
            info['M_stars'] = gx.stars['mass'].sum().in_units_of('Msol', subs=self.__snapshot)
            info['M_gas'] = gx.gas['mass'].sum().in_units_of('Msol', subs=self.__snapshot)
            info['Z_stars'] = (gx.stars['mass'] * gx.stars['metallicity']).sum() / gx.stars['mass'].sum() \
                                    if len(gx.stars) else None
            info['Z_gas'] =  (gx.gas['mass'] * gx.gas['metallicity']).sum() / gx.gas['mass'].sum()
            info['SFR'] = gx.gas['sfr'].sum()
            info['jzjc_baryons'] = gx.baryons['jzjc'].mean()
            info['jzjc_total'] = gx['jzjc'].mean()
            info['L_baryons'] = gx.baryons['angmom'].sum(axis=0)
            redI = pg.UnitArr(pg.analysis.reduced_inertia_tensor(gx)).flatten()
            info['I_red'] = redI
            info['R_half_M'] = pg.analysis.half_mass_radius(gx)
            info['R_half_M_stellar'] = pg.analysis.half_mass_radius(gx.stars)
        else:
            info['M_stars'] =  pg.UnitArr(0, 'Msol')
            info['M_gas'] = pg.UnitArr(0, 'Msol')
            info['Z_stars'] = None
            info['Z_gas'] = None
            info['SFR'] = None
            info['L_baryons'] = None
            info['I_red'] = None
            info['R_half_M'] = None

        # infos part 3
        disc_def = dict()  # jzjc_min=0.85, rmax='50 kpc', zmax='10 kpc')
        # pg.analysis.orientate_at(gx, 'red I', total=True)
        disc = gx[pg.DiscMask(**disc_def)]
        if gx:
            info['R_half_M(faceon)'] = pg.analysis.half_qty_radius(gx, 'mass', proj=2)
            info['R_eff(faceon)'] = pg.analysis.eff_radius(gx)
            info['D/T(stars)'] = float(disc.stars['mass'].sum() / gx.stars['mass'].sum())
        else:
            info['R_half_M(faceon)'] = None
            info['R_eff(faceon)'] = None
            info['D/T(stars)'] = None

        self.__gx_properties = info
        return


    def _center_gx(self, center_known):
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("center on galaxy...")
        stars = self.__snapshot.stars
        if center_known is None:
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("searching for galaxies - shrinking sphere...")
            center = pg.analysis.shrinking_sphere(stars, center=[stars.boxsize / 2] * 3, R=stars.boxsize)
        else:
            center = center_known

        self.__center = center
        pg.Translation(-center).apply(self.__snapshot, total=True)

        vel_center = pg.analysis.mass_weighted_mean(self.__snapshot[self.__snapshot['r'] < '1 kpc'], 'vel')
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('center velocities at:', vel_center)
        self.__snapshot['vel'] -= vel_center

        # Rvir_property = self.__snap_config.Rvir_property
        # Rvir = self.dm_halo.props[Rvir_property]
        Rgx = self.Rvir * self.gx_radius
        #Rgx_units = pg.UnitScalar(Rgx).in_units_of(self.__snapshot['r'].units)
        #mask = pg.BallMask(str(Rgx) + ' ckpc h_0**-1', center=[0, 0 , 0])
        #mask = pg.BallMask(pg.UnitScalar(Rgx).in_units_of(self.snapshot['r'].units), center=[0, 0 , 0])
        mask = pg.BallMask(Rgx, center=[0, 0 , 0])
        snapshot_gx = self.__snapshot[mask]

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("center on galaxy - complete")
            print("*********************************************")

        self.__galaxy = snapshot_gx
        self.__galaxy_all = self.__galaxy
        return


    def _gx_orientate(self):
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("orientate galaxy...")

        #pg.analysis.orientate_at(self.__galaxy_all, 'L', total=True)
        if 'I_red' in self.__gx_properties:
            # redI = self.__gx_properties['I_red']
            redI = pg.UnitArr(pg.analysis.reduced_inertia_tensor(self.__galaxy_all)).flatten()
            if redI is not None:
                redI = redI.reshape((3, 3))
                mode, qty = 'red I', redI
                pg.analysis.orientate_at(self.__galaxy_all, mode, qty=qty, total=True)
            else:
                pg.analysis.orientate_at(self.__galaxy_all, 'L', total=True)
        else:
            pg.analysis.orientate_at(self.__galaxy_all, 'L', total=True)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("orientate complete")
            print("*********************************************")
        return

    def _prepare_zoom(self, mode='auto', shrink_on='stars', ret_FoF=False,
                     sph_overlap_mask=False, star_form='deduce',
                     gas_trace='deduce', to_physical=True, load_double_prec=False,
                     fill_undefined_nan=True, gas_traced_blocks='all',
                     gas_traced_dervied_blocks=None, **kwargs):
        '''
        A convenience function to load a snapshot from a zoomed-in simulation that is
        not yet centered or oriented.

        Args:
            self (str, Snap):   The snapshot of a zoomed-in simulation to prepare.
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
            info (str, dict):   Now gx_properties
                                Path to info file or the dictionary as returned from
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

            not clear how to use these parameters in first release, will be moved to cCache-commands
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
            if 'lowres_threshold' in self.__profile_properties:
                threshold = self.__profile_properties['lowres_threshold']
            else:
                threshold = pg.environment.DEFAULT_lowres_threshold

            M_lowres = h.lowres_mass
            M = h.mass
            return M_lowres / M > threshold

        # if 'gastrace' in kwargs:
        #     raise ValueError("You passed 'gastrace'. Did you mean 'gas_trace'?")
        # if 'starform' in kwargs:
        #     raise ValueError("You passed 'starform'. Did you mean 'star_form'?")

        s = self.__snapshot

        # if isinstance(s, str):
        #     s = Snapshot(s, load_double_prec=load_double_prec)
        gal_R200 = self.__profile_properties['gx_radius']
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('prepare zoomed-in', s)

        # read info file (if required)
        if mode in ['auto', 'info']:
            if len(self.__gx_properties) == 0:
                if mode == 'auto':
                    mode = 'FoF'
                else:
                    raise IOError('Could not read/find the info file (gx.info)!')
            else:
                if mode == 'auto':
                    mode = 'info'

        # if to_physical:
        #     s.to_physical_units()

        # find center
        if mode == 'info':      # info file found an loaded into gx_properties
            center = self.__gx_properties['center']
        elif mode in ['ssc', 'FoF']:
            # default old parameter linking_length=None
            if 'linking_length' in self.__profile_properties:
                linking_length = self.__profile_properties['linking_length']
            else:
                linking_length = pg.environment.DEFAULT_linking_length

            # default old parameter linking_vel='200 km/s',
            if 'linking_vel' in self.__profile_properties:
                linking_vel = self.__profile_properties['linking_vel']
            else:
                linking_vel = pg.environment.DEFAULT_linking_vel

            if mode == 'FoF':
                try:
                    halos = pg.analysis.generate_FoF_catalogue(
                        s,
                        l=linking_length,
                        exclude=FoF_exclude,
                        #calc=['mass', 'lowres_mass'],       # generate all halo properties
                        max_halos=1,                         #  take only 3 instead of 10, for performance
                        progressbar=False,
                        **kwargs
                    )
                except Exception as e:
                    print("error finding halo ", e)
                    halos = None
                    halo = None
                    gal = None
                    if ret_FoF:
                        return halo, gal, halos
                    else:
                        return halo, gal

                if shrink_on not in ['all', 'highres']:     # $$ shrink_on parameter nicht klar
                    try:
                        galaxies = pg.analysis.generate_FoF_catalogue(
                            get_shrink_on_sub(s, shrink_on),
                            l=linking_length,
                            dvmax=linking_vel,
                            calc=['mass', 'com'],
                            max_halos=5,                    # take only 5 instead of 10
                            progressbar=False,
                            **kwargs
                        )
                        # The most massive galaxy does not have to be in a halo with little
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
                    except Exception as e:
                        galaxy = None
                        print("Error finding galaxy ", e)

                    if galaxy is None:
                        shrink_on = None
                    else:
                        shrink_on = s[galaxy]
                else:
                    shrink_on = s[halos[0]]
            elif mode == 'ssc':
                shrink_on = get_shrink_on_sub(s, shrink_on)

            if shrink_on is not None and len(shrink_on) > 0:
                com = pg.analysis.center_of_mass(s)
                R = np.max(pg.utils.periodic_distance_to(s['pos'], com, s.boxsize))
                center = pg.analysis.shrinking_sphere(shrink_on, com, R)
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
            pg.Translation(-center).apply(s, total=True)
            self.__center = center
            # center the velocities

        # cut the halo (<R200)
        if mode == 'info':
            R200 = self.__gx_properties['center']['R200']
            M200 = self.__gx_properties['center']['M200']
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('derive virial information')
            R200, M200 = pg.analysis.virial_info(s)
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('R200:', R200)
            print('M200:', M200)
        halo = s[pg.BallMask(R200, sph_overlap=sph_overlap_mask)]

        # orientate at the reduced inertia tensor of the baryons wihtin 10 kpc
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('orientate', end=' ')
        if mode == 'info':
            if 'I_red' in self.__gx_properties:
                redI = self.__gx_properties['I_red']
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
                mode, qty = 'vec', self.__gx_properties['L_baryons']
            s.orientate_at(s, mode, qty=qty, total=True)
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('at red. inertia tensor of the baryons within %.3f*R200' % gal_R200)
            pg.analysis.orientate_at(s[pg.BallMask(gal_R200 * R200, sph_overlap=False)].baryons,
                         'red I',
                         total=True
                         )

        # cut the inner part as the galaxy
        gal = s[pg.BallMask(gal_R200 * R200, sph_overlap=sph_overlap_mask)]
        Ms = gal.stars['mass'].sum()
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('M*:  ', Ms)

        if len(gal) == 0:
            gal = None
        if len(halo) == 0:
            halo = None

        # these functions will be moved to gCache-commands (former gtracegas, gtrace)
        # if star_form == 'deduce':
        #     try:
        #         star_form = os.path.dirname(s.filename) + '/trace/star_form.ascii'
        #     except:
        #         print('WARNING: could not deduce the path to the ' + \
        #               'star formation file!', file=sys.stderr)
        #         star_form = None
        # if isinstance(star_form, str):
        #     star_form = os.path.expanduser(star_form)
        #     if not os.path.exists(star_form):
        #         print('WARNING: There is no star formation file ' + \
        #               'named "%s"' % star_form, file=sys.stderr)
        #         star_form = None
        #     else:
        #         if environment.verbose >= environment.VERBOSE_NORMAL:
        #             print('read star formation file from:', star_form)
        #         fill_star_from_info(s, star_form,
        #                             fill_undefined_nan=fill_undefined_nan)
        #
        # if gas_trace == 'deduce':
        #     try:
        #         directory = os.path.dirname(s.filename) + '/../'
        #         candidates = []
        #         for fname in os.listdir(directory):
        #             if fname.startswith('gastrace'):
        #                 candidates.append(fname)
        #         if len(candidates) == 1:
        #             gas_trace = directory + candidates[0]
        #         else:
        #             raise RuntimeError('too many candidates!')
        #     except:
        #         print('WARNING: could not deduce the path to the ' + \
        #               'gas tracing file!', file=sys.stderr)
        #         gas_trace = None
        # if isinstance(gas_trace, str):
        #     gas_trace = os.path.expanduser(gas_trace)
        #     if not os.path.exists(gas_trace):
        #         print('WARNING: There is no gas trace file named ' + \
        #               '"%s"' % gas_trace, file=sys.stderr)
        #         gas_trace = None
        #     else:
        #         if environment.verbose >= environment.VERBOSE_NORMAL:
        #             print('read gas trace file from:', gas_trace)
        #         if gas_traced_dervied_blocks is None:
        #             gas_traced_dervied_blocks = (gas_traced_blocks == 'all')
        #         fill_gas_from_traced(s, gas_trace,
        #                              add_blocks=gas_traced_blocks,
        #                              add_derived=gas_traced_dervied_blocks)

        if mode == 'FoF' and ret_FoF:
            return halo, gal, halos
        else:
            return  halo, gal



    def _fill_star_from_info(self, fname, a_max, fill_undefined_nan=True, dtypes=None,
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
        snap = self.__snapshot
        stars = snap.root.stars
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
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
        try:
            # load the data
            SFI = np.loadtxt(fname, skiprows=1, dtype=dtypes)
            if a_max < 1.0:
                SFI = SFI[SFI['aform'] <= a_max]

            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('testing if the IDs match the (root) snapshot...')
            SFI_IDs = SFI['ID']
            # test uniqueness removed
            # if not stars.IDs_unique():
            #    raise RuntimeError('Stellar IDs in the snapshot are not unique!')
            # if len(np.unique(SFI_IDs)) != len(SFI_IDs):
            #    raise RuntimeError('IDs in the star formation file are not unique!')
            # there might be too many or not enough IDs in the file

            missing = np.setdiff1d(stars['ID'], SFI_IDs, assume_unique=False)
            if fill_undefined_nan:
                if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                    print('WARNING: There are %d stellar IDs missing ' % len(missing) + \
                      'in the formation file! Fill them with NaN\'s.')
                add = np.array([(ID,) + (np.NaN,) * (len(dtypes) - 1) for ID in missing],
                               dtype=dtypes)
                SFI = np.concatenate((SFI, add))
                del add
                SFI_IDs = SFI['ID']
            else:
                raise RuntimeError('Some stars do not have a match in the ' + \
                                   'formation file "%s" (missing: %d)!' % (fname, len(missing))
                                   )
            toomany = np.setdiff1d(SFI_IDs, stars['ID'], assume_unique=False)
            if len(toomany):
                raise RuntimeError('Some formation file IDs do not have a match in ' + \
                                   'the snapshot (missing: %d)!' % (
                                       len(toomany))
                                   )

            if len(SFI) != len(stars):
                raise RuntimeError('Different number of stars in snapshot (%d) ' + \
                                   'and the formation file (%d)!' % (len(stars), len(SFI)))

            # adding the data as blocks
            sfiididx = np.argsort(SFI_IDs)
            sididx = np.argsort(stars['ID'])
            for name in dtypes.names:
                if name == 'ID':
                    continue
                if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                    print('adding the new block "%s" (units:%s, type:%s)...' % (
                        name, units.get(name, None), dtypes[name]))
                stars[name] = pg.UnitArr(np.empty(len(stars)),
                                         dtype=dtypes[name],
                                         units=units.get(name, None))
                stars[name][sididx] = SFI[name][sfiididx]
            return

        except Exception as e:
            print("error importing star_form file")
            print(e)
            exit(1)
            return

    def _fill_gas_from_traced(self, snap, data, add_blocks='all', add_derived=True,
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
            if pg.environment.verbose >= pg.environment.VERBOSE_TALKY:
                print('read gas trace file "%s"...' % filename)
                sys.stdout.flush()
            import pickle as pickle
            with open(filename, 'rb') as f:
                tr = pickle.load(f)
                f.close()
            return tr

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
            if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                print('adding blocks that can be derived from the gas trace blocks:')

            gas = snap.gas

            trmask = (gas['num_recycled'] != -1)
            set_N_cycles = set(gas['num_recycled'])
            max_N_cycle = max(set_N_cycles)

            # each (full) cycle takes some given time
            if max_N_cycle > 0:
                if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                    print('  "out_time",')
                ejected = gas['ejection_time'][:, :-1]
                infall = gas['infall_time'][:, 1:]
                gas['out_time'] = infall - ejected
                mask = (ejected == invalid) | ~np.isfinite(ejected)
                del ejected
                pg.environment.gc_full_collect()
                gas['out_time'][mask] = invalid
                mask = (infall == invalid) | ~np.isfinite(infall)
                del infall
                pg.environment.gc_full_collect()
                gas['out_time'][mask] = invalid
            pg.environment.gc_full_collect()

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
                if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                    print('  "%s"' % last)
                gas[last] = pg.UnitArr(np.empty(len(gas), dtype=gas[alle].dtype),
                                       units=gas[alle].units)
                gas[last][~trmask] = invalid
                gas[last][trmask] = gas[alle][last_infall_idx][trmask]
            del last_infall_idx
            pg.environment.gc_full_collect()

            # last_ejection_idx = np.sum(~np.isnan(gas['ejection_a']), axis=-1) - 1
            last_ejection_idx = np.sum(gas['ejection_a'] != invalid, axis=-1) - 1
            last_ejection_idx = np.arange(len(last_ejection_idx)), last_ejection_idx
            for last, alle in [('last_ejection_a', 'ejection_a'),
                               ('last_ejection_time', 'ejection_time'),
                               ('mass_at_last_ejection', 'mass_at_ejection'),
                               ('metals_at_last_ejection', 'metals_at_ejection'),
                               ('jz_at_last_ejection', 'jz_at_ejection'),
                               ('T_at_last_ejection', 'T_at_ejection')]:
                if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                    print('  "%s"' % last)
                gas[last] = pg.UnitArr(np.empty(len(gas), dtype=gas[alle].dtype),
                                       units=gas[alle].units)
                gas[last][~trmask] = invalid
                gas[last][trmask] = gas[alle][last_ejection_idx][trmask]
            del last_ejection_idx
            pg.environment.gc_full_collect()
            return

        def add_block(name, block):
            if name in add_blocks:
                if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
                    print('  "%s"' % name)
                gas[name] = block
            return

        def count_cycles(history):
            c = 0
            for entry in history:
                if entry[0] == 'r':  # reentry
                    c += 1
            return c

        def get_a_at(event, history, max_count, invalid):
            a = []
            n = 0
            for entry in history:
                if entry[0] == event:
                    a.append(entry[1][0])  # a at event
                    n += 1
            while n < max_count:
                a.append(invalid)
                n += 1

            return a

        def get_event_Q(event, history, index_Q, max_count, invalid):
            Q = []
            n = 0
            for entry in history:
                if entry[0] == event:
                    Q.append(entry[1][index_Q])  # a at reentry
                    n += 1
            while n < max_count:
                Q.append(invalid)
                n += 1

            return Q

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
        pg.environment.gc_full_collect()

        if isinstance(data, str):
            filename = data
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('reading the gas trace information from %s...' % filename)
            tr = read_traced_gas(data)
        else:
            filename = '<given data>'

        # filter to gas only (last history entry: new gas or reentered gas)
        # gas_type = {1, 2}
        # data = dict([i for i in iter(data.items()) if i[1][0][0] in gas_type])
        data = dict([i for i in tr.items() if i[1][len(i[1]) - 1][0] in ['n', 'r']])

        # print("root parts", snap.root.parts)
        # if len(set(data.keys()) - set(snap.root['ID'])) > 0: print("nicht in gesamt")
        # if len(set(data.keys()) - set(snap.root.stars['ID'])) > 0: print("nicht in stars")
        # if len(set(data.keys()) - set(snap.root.gas['ID'])) > 0: print("nicht in gas")

        allIDs = set(snap.root['ID'])
        todel = []
        for pk in data.keys():
            if not pk in allIDs:
                todel.append(pk)
        for pk in todel:
            data.pop(pk)

        if len(set(data.keys()) - set(snap.root['ID'])) > 0: print("nicht in gesamt")

        if pg.environment.verbose >= pg.environment.VERBOSE_TALKY:
            print('test IDs and find matching IDs...')
        if len(set(data.keys()) - set(gas['ID'])) > 0:
            raise RuntimeError('Traced gas IDs in "%s" have ' % filename +
                               '%s ' % pg.utils.nice_big_num_str(len(
                set(data.keys()) - set(gas['ID']))) +
                               'elements that are not in the snapshot!')
        tracedIDs = (set(data.keys()) & set(gas['ID']))

        trmask = np.array([(ID in tracedIDs) for ID in gas['ID']], dtype=bool)
        if pg.environment.verbose >= pg.environment.VERBOSE_TALKY:
            print('  found %s (of %s)' % (pg.utils.nice_big_num_str(len(tracedIDs)),
                                          pg.utils.nice_big_num_str(len(data))), end=' ')
            print('traced IDs that are in the snapshot')

        if pg.environment.verbose >= pg.environment.VERBOSE_NORMAL:
            print('adding the blocks:')

        trididx = np.argsort(list(data.keys()))
        gididx = np.argsort(gas['ID'])
        gididx_traced = gididx[trmask[gididx]]

        # type = not traced (0), in region (1), out of region (2)
        # number of full cycles (out and in)

        trace_type = pg.UnitArr(np.zeros(len(gas), dtype=int))
        trtype = np.array([1 if i[len(i) - 1][0] == 'n' else 2 for i in data.values()])
        trace_type[gididx_traced] = trtype[trididx]
        add_block('trace_type', trace_type)
        del trace_type

        n_cyc = np.array([count_cycles(i) for i in data.values()])
        num_recycled = pg.UnitArr(np.empty(len(gas), dtype=n_cyc.dtype))
        num_recycled[~trmask] = -1
        num_recycled[gididx_traced] = n_cyc[trididx]
        add_block('num_recycled', num_recycled)
        set_N_cycles = set(n_cyc)
        if pg.environment.verbose >= pg.environment.VERBOSE_TALKY:
            print('  +++ number of recycles that occured:', set_N_cycles, '+++')

        # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
        # events of entering the region. N is the number of the gas particles. Each
        # particle has entries for each of their enter events, those events that do
        # not exists, get filled with `invalid`.

        max_N_cycle = max(set_N_cycles)
        # infall_t = np.array([[e[0] for e in i[1:1 + 3 * n + 1:3]] + [invalid] * (max_N_cycle - n)
        #                     for n, i in zip(n_cyc, list(data.values()))])
        infall_t = np.array([get_a_at('r', i, max_N_cycle + 1, invalid) for n, i in zip(n_cyc, list(data.values()))])
        infall_a = pg.UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                       dtype=infall_t.dtype), units=units['TIME'])
        infall_a[~trmask] = invalid
        infall_a[gididx_traced] = infall_t[trididx]
        add_block('infall_a', infall_a)
        del infall_t
        pg.environment.gc_full_collect()

        # from .snapshot import age_from_form
        # only convert reasonable values & ensure not to overwrite blocks
        mask = (infall_a != invalid) & np.isfinite(infall_a)
        infall_time = infall_a.copy()
        new = gas.cosmic_time() - pg.age_from_form(infall_time[mask], subs=gas)
        infall_time.units = new.units
        infall_time[mask] = new
        del new
        add_block('infall_time', infall_time)
        del infall_time
        pg.environment.gc_full_collect()

        for name, idx, unit in [('mass_at_infall', 1, units['MASS']),
                                ('metals_at_infall', 2, units['MASS']),
                                ('jz_at_infall', 3, units['ANGMOM']),
                                ('T_at_infall', 4, units['TEMP'])]:
            if name not in add_blocks:
                continue
            # infall_Q = np.array([[e[idx] for e in i[1:1 + 3 * n + 1:3]] +
            #                     [invalid] * (max_N_cycle - n)
            #                     for n, i in zip(n_cyc, list(data.values()))])
            infall_Q = np.array(
                [get_event_Q('r', i, idx, max_N_cycle + 1, invalid) for n, i in zip(n_cyc, list(data.values()))])
            block = pg.UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                        dtype=infall_Q.dtype),
                               units=unit)
            block[~trmask] = invalid
            block[gididx_traced] = infall_Q[trididx]
            add_block(name, block)
            del infall_Q, block
            pg.environment.gc_full_collect()

        # Create blocks with shape (N, max(recycl)+1) for traced quantities at the
        # events of ejection / leaving the region. N is the number of the gas
        # particles. Each particle has entries for each of their ejection events,
        # those events that do not exists, get filled with `invalid`.
        max_N_cycle = max(set_N_cycles)
        # eject_t = np.array([[e[0] for e in i[2:2 + 3 * n:3]] +
        #                     [i[2 + 3 * n][0] if t == 2 else invalid] +
        #                     [invalid] * (max_N_cycle - n)
        #                     for n, t, i in zip(n_cyc, trtype, list(data.values()))])
        eject_t = np.array([get_a_at('l', i, max_N_cycle + 1, invalid) for n, i in zip(n_cyc, list(data.values()))])

        ejection_a = pg.UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                         dtype=eject_t.dtype),
                                units=units['TIME'])
        ejection_a[~trmask] = invalid
        ejection_a[gididx_traced] = eject_t[trididx]
        del eject_t
        add_block('ejection_a', ejection_a)
        pg.environment.gc_full_collect()

        # from .snapshot import age_from_form
        # only convert reasonable values & ensure not to overwrite blocks
        mask = (ejection_a != invalid) & np.isfinite(ejection_a)
        ejection_time = ejection_a.copy()
        new = gas.cosmic_time() - pg.age_from_form(ejection_time[mask], subs=gas)
        ejection_time.units = new.units
        ejection_time[mask] = new
        del new
        add_block('ejection_time', ejection_time)
        pg.environment.gc_full_collect()

        for name, idx, unit in [('mass_at_ejection', 1, units['MASS']),
                                ('metals_at_ejection', 2, units['MASS']),
                                ('jz_at_ejection', 3, units['ANGMOM']),
                                ('T_at_ejection', 4, units['TEMP'])]:
            if name not in add_blocks:
                continue
            # eject_Q = np.array([[e[idx] for e in i[2:2 + 3 * n:3]] +
            #                     [i[2 + 3 * n][idx] if t == 2 else invalid] +
            #                     [invalid] * (max_N_cycle - n)
            #                     for n, t, i in zip(n_cyc, trtype, list(data.values()))])
            eject_Q = np.array(
                [get_event_Q('l', i, idx, max_N_cycle + 1, invalid) for n, i in zip(n_cyc, list(data.values()))])
            block = pg.UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                        dtype=eject_Q.dtype),
                               units=unit)
            block[~trmask] = invalid
            block[gididx_traced] = eject_Q[trididx]
            add_block(name, block)
            del eject_Q, block
            pg.environment.gc_full_collect()

        # for each cycle there is a maximum travel distance, plus one more for those
        # particles that are outside the region: store them
        for name, idx, unit in [('cycle_r_max_at', 0, units['TIME']),
                                ('cycle_r_max', 1, units['POS']),
                                ('cycle_z_max_at', 2, units['TIME']),
                                ('cycle_z_max', 3, units['POS'])]:
            if name not in add_blocks:
                continue
            # pos = np.array([[e[idx] for e in i[3:3 + 3 * n:3]] +
            #                 [i[3 + 3 * n][idx] if t == 2 else invalid] +
            #                 [invalid] * (max_N_cycle - n)
            #                 for n, t, i in zip(n_cyc, trtype, list(data.values()))])
            pos = np.array(
                [get_event_Q('p', i, idx, max_N_cycle + 1, invalid) for n, i in zip(n_cyc, list(data.values()))])
            block = pg.UnitArr(np.empty((len(gas), max_N_cycle + 1),
                                        dtype=pos.dtype),
                               units=unit)
            block[~trmask] = invalid
            block[gididx_traced] = pos[trididx]
            add_block(name, block)
            del pos, block
            pg.environment.gc_full_collect()

        if add_derived:
            fill_derived_gas_trace_qty(snap, units=units, invalid=invalid)
        return

    def print_results(self):
        print(" ")
        print("**********************************************************")
        print(("Summary of results for snapfile ", self.__data_file))
        print("**********************************************************")
        print(("number of results: ", len(self.results)))
        print("--------------------------------------------------------------")
        for result in self.results:
            print(("result:", result, "=", self.results[result]))
        print("--------------------------------------------------------------")
        return

class SnapshotProperty(dict):

    def __init__(self):
        dict.__init__(self)
        self.__modified = False
        self.__info_filename = ''

    def append(self, name, value):
        if name in self:
            v = self[name]
            if isinstance(v, pg.UnitArr) or isinstance(value, pg.UnitArr) or v != value:
                self[name] = value
                self.__modified = True
        else:
            v = None
            self[name] = value
            self.__modified = True
        return v        # returns old value

    def remove(self, name):
        if name in self:
            v = self[name]
            v = self.pop(name, None)
            self.__modified = True
        else:
            v = None
        return v

    @property
    def modified(self):
        return self.__modified


    def _write_info_file(self, info_filename, force=False):

        self.__info_filename = info_filename
        if not self.__modified and not force:
            return

        if os.path.exists(info_filename):
            os.remove(info_filename)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print('start writing SnapshotProperty file...')
            print("*********************************************")

        with open(info_filename, 'w') as f:
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('file generated by gadutils.SnapshotFile', file=f)
            print('', file=f)
            infos = self
            for prop in infos:
                value = str(infos[prop]).replace('\n',' ')
                if isinstance(infos[prop], str):
                    value = str(infos[prop]).replace('\n', ' ')
                    value = '"' + value + '"'
                    print(prop, ' : ', value, file=f)
                elif isinstance(infos[prop], io.BytesIO):
                    buf = infos[prop]
                    buf.seek(0)
                    im = Image.open(buf)
                    save_parts = os.path.split(info_filename)
                    save_property = save_parts[1] + '.' + prop + '.png'
                    save_filename = save_parts[0] + '/' + save_property
                    im.save(save_filename)
                    buf.close()
                    print(prop, ' : ', '$' + save_property, file=f)
                else:
                    value = str(infos[prop]).replace('\n', ' ')
                    print(prop, ' : ', value, file=f)
            print('', file=f)
            f.close()
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print('writing SnapshotProperty file complete')
            print("*********************************************")

        self.__modified = False
        return

    def get_binary_value(self, property):

        try:
            prop = self[property]
            if prop[0] != '$':
                return None
            prop = prop.strip('$')
            save_parts = os.path.split(self.__info_filename)
            save_filename = save_parts[0] + '/' + prop
            with open(save_filename, 'rb') as fin:
                data = io.BytesIO(fin.read())
                fin.close()
                return data
        except Exception as e:
            pass

        return None

    def load_binary_properties(self):
        loaded = 0
        for prop in self:
            value = self[prop]
            if isinstance(value,str) and value[0] == '$':
                bin_value = self.get_binary_value(prop)
                self[prop] = bin_value
                loaded += 1
        return loaded

    def _read_info_file(self, info_filename):
        '''
        Read in the contents of an info file as produced by gtrace into a dictionary.

        It is assumed, that there is exactly one colon per line, which separates the
        name (the part before the colon) and the value (the part after the colon). For
        the value part the following is assumed: if there are no brackets ('[') it is
        a float; if there is one pair of brackets, it is a float with units, which are
        given within the brackets; otherwise the value is simply stored as a string.

        Args:
            info_filename (str):     The path to the info file.

        Returns:
            info (dict):        A dictionary containing all the entries from the file.
        '''
        self.__info_filename = info_filename
        self.clear()
        info = self
        with open(os.path.expanduser(info_filename), 'r') as finfo:
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("*********************************************")
                print("reading SnapshotProperty from file ", info_filename)
                print("*********************************************")
            for line in finfo:
                try:
                    cols = line.split(':', 1)
                    if len(cols) > 1:  # otherwise no value given, avoid ValueError
                        name = cols[0].strip()
                        value = cols[1].strip()
                        blocks = re.findall('\[(.*?)\]', value)
                        if len(blocks) == 1:
                            valuelist = len(value.rsplit('[')[0].strip()) == 0
                        else:
                            valuelist = False
                        if len(blocks) == 0:
                            if value[0] == '"':
                                value = value.strip('"')
                            elif value[0] == '$':
                                value = value.strip('"')
                            else:
                                if value == 'None':
                                    value = None
                                elif value == 'True':
                                    value = True
                                elif value == 'False':
                                    value = False
                                else:
                                    value = float(value)
                        elif len(blocks) == 1 and not valuelist:
                            value = pg.UnitArr(float(value.rsplit('[')[0].strip()),
                                            units=blocks[0])
                        elif len(blocks) == 1 and valuelist:
                            value = pg.UnitArr([int(s) if len(s.split('.')) == 1 else float(s) for s in value.strip('[]').split()],
                                               units=None)
                        elif len(blocks) == 2:
                            value = pg.UnitArr([float(s.strip(', ')) for s in blocks[0].split()],
                                            units=None if blocks[1] == 'None' else blocks[1])
                        if name in info:
                            print('WARNING: "%s" occures ' % name + \
                                  'multiple times in info file ' + \
                                  '"%s"! First one used.' % info_filename, file=sys.stderr)
                        info[name] = value
                except ValueError as e:
                    continue  # ignore error continue loading
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("SnapshotProperty")
            for prop in info:
                v = info[prop]
                if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                    print("......", prop, " = ", v)

            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("*********************************************")
                print("reading complete ")
                print("*********************************************")

        self.__modified = False
        return self
