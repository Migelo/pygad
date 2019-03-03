import os
import io
from PIL import Image
import sys
import re
import time
import pygad as pg            # use absolute package loading
from .. import environment

__all__ = ["SnapshotCache", "SnapshotProperty"]

class SnapshotCache:
    def __init__(self, snapfile_name, profile='cache'):
        # type: (str, str) -> SnapshotCache
        pg.environment.verbose = environment.verbose
        self.__data_file = snapfile_name
        self.__profile = ''
        self.__profile_properties = SnapshotProperty()
        self.__load_profile(profile)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("************************************************************************")
            print("using snapfile ",self.__data_file)
            print("************************************************************************")

        self.__results = dict()
        self.__data = None
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
    def dm_halo(self):
        if self.__dm_halo is None:
            self.__create_DM_halo()
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
        return self.__data

    @snapshot.setter
    def snapshot(self, value):
        self.__data = value

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
        Rvir_f  = float(self.__halo_properties[Rvir_property])
        return Rvir_f

    @property
    def gx_radius(self):
        gx_rad = self.__profile_properties['gx_radius']
        return gx_rad

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
        postfix_profile = self.__filename + '-profile.info'
        return self.__destination + "/" + postfix_profile

    def get_profile_path(self):
        profile_path = self.__destination + '/'
        return profile_path

    def load_snapshot(self, useChache=True, forceCache=False, loadBinaryProperties=False):
        # type: (bool, bool) -> pg.snapshot
        filename = self.name
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print(("loading snapshot " + filename + "..."))
            print("*********************************************")
        #snapshot = pg.Snap(filename, load_double_prec=True)
        snapshot = pg.Snapshot(filename, load_double_prec=True)
        self.__data = snapshot

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("boxsize ", snapshot.boxsize)
            print("particles ", snapshot.parts)
            print("headers")
            for h in snapshot.headers()[0]:
                print("......", h, " = ", snapshot.headers()[0][h])
            print("units", snapshot.gadget_units)

        try:
            if not useChache or forceCache:
                raise FileNotFoundError
            self.__halo_properties._read_info_file(self.__get_halo_filename())
            self.__center_gx()
            self.__gx_orientate()
            self.__gx_properties._read_info_file(self.__get_gx_filename())
            if loadBinaryProperties:        # only when reloading, as no binary data in generated automatically
                self.halo_properties.load_binary_properties()
                self.gx_properties.load_binary_properties()
                self.profile_properties.load_binary_properties()
        except Exception as e:
            self.__create_DM_halo()
            if useChache or forceCache:
                self.__halo_properties._write_info_file(self.__get_halo_filename(), force=forceCache)
            self.__center_gx()
            self.__gx_orientate()
            self.__create_gx_properties()
            if useChache or forceCache:
                self.__gx_properties._write_info_file(self.__get_gx_filename(), force=forceCache)
                self.__profile_properties._write_info_file(self.__get_profile_filename(), force=forceCache)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("loading complete Rvir=", self.Rvir)
            print("*********************************************")
        self.__loaded = True
        return self.__data

    def write_chache(self, forceCache = False):
        filename = self.name
        self.__halo_properties._write_info_file(self.__get_halo_filename(), force=forceCache)
        self.__gx_properties._write_info_file(self.__get_gx_filename(), force=forceCache)
        self.__profile_properties._write_info_file(self.__get_profile_filename(), force=forceCache)
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
            self.load_snapshot()
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print('*** loaded')
        #dfSnap = self
        globalsParameter = globals().copy()
        globalsParameter['np'] = np
        globalsParameter['plt'] = plt

        localsParameter = locals().copy()
        localsParameter['dfSnap'] = self

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
        self.write_chache()
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** finished')

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print('*** close')
        localsParameter['snap_exec'] = 'close'
        if command_str != '':
            exec(command_str, globalsParameter, localsParameter)

        return

    def __load_profile(self, profile):
        if profile == '':
            profile_name = 'cache'
        else:
            profile_name = profile

        split_names = os.path.split(self.name)
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

        return

    def __create_DM_halo(self):
        def convert_to_Units(prop):
            if isinstance(prop,tuple):
                value = pg.UnitArr([s for s in prop], units=None)
                return value
            else:
                return prop

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print(("creating DM-halo ", self.__data.parts))
            print("*********************************************")

        search_snap = self.__data.dm

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("search_snap ", search_snap.parts)

        FoF, N_FoF = pg.analysis.find_FoF_groups(search_snap, l='6 ckpc')
        halos = pg.analysis.generate_FoF_catalogue(search_snap, max_halos=1, FoF=FoF,
                                                       exlude=lambda h, s: h.lowres_mass / h.mass > 0.01)
        dm_halo = halos[0]  # most massive halo

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("DM-halo center ", dm_halo.ssc)
            print("DM-halo properties")

        halo_properties = SnapshotProperty()
        halo_properties.append("date-created", '"' + time.asctime( time.localtime(time.time()) ) + '"')
        halo_properties.append("profile-name", self.__profile)
        for prop in dm_halo.props:
            v = convert_to_Units(dm_halo.props[prop])
            halo_properties[prop] = v
            if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
                print("......", prop, " = ", v)

        self.__dm_halo = dm_halo
        self.__halo_properties = halo_properties

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("creating complete ")
            print("*********************************************")
        return dm_halo

    def __create_gx_properties(self):

        info = SnapshotProperty()
        info.append("date-created", '"' + time.asctime( time.localtime(time.time()) ) + '"')
        info.append("profile-name", self.__profile)
        gx = self.__galaxy_all
        halo = self.__data

        R200 = float(self.halo_properties['R200_com'])
        M200 = float(self.halo_properties['M200_com'])

        R500 = float(self.halo_properties['R500_com'])
        M500 = float(self.halo_properties['M500_com'])

        # infos part 1
        info['a'] = self.__data.scale_factor
        info['cosmic_time'] = self.__data.cosmic_time()
        info['redshift'] = self.__data.redshift
        info['h'] = self.__data.cosmology.h(self.__data.redshift)
        #            print('center:             ', center, file=f)
        info['center_com'] = pg.analysis.center_of_mass(gx) if gx is not None else None
        info['center_ssc)'] = pg.analysis.shrinking_sphere(gx,
                                                                   center=self.__center,
                                                                   R=R200) if gx is not None else None
        info['center_halo_com)'] = pg.analysis.center_of_mass(halo) if halo is not None else None
        info['center_halo_ssc)'] = pg.analysis.shrinking_sphere(halo,
                                                                   center=self.__center,
                                                                   R=R200) if halo is not None else None
        info['R200'] = R200
        info['M200'] = M200
        info['R500'] = R500
        info['M500'] = M500

        # infos part 2
        # h = self.__data[pg.BallMask(R200, center=self.__center, sph_overlap=False)]
        # h=halo?
        if halo:
            info['M_stars'] = halo.stars['mass'].sum().in_units_of('Msol', subs=self.__data) \
                                        if len(halo.stars) else None
            info['L_halo'] = halo['angmom'].sum(axis=0)
            info['L_baryons'] = halo.baryons['angmom'].sum(axis=0)
        else:
            info['M_stars'] = pg.UnitArr(0, units='Msol')
            info['L_halo'] = None
            info['L_baryons'] = None
        if gx:
            info['M_stars'] = gx.stars['mass'].sum().in_units_of('Msol', subs=self.__data)
            info['M_gas'] = gx.gas['mass'].sum().in_units_of('Msol', subs=self.__data)
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


    def __center_gx(self):
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("searching for galaxies - shrinking sphere...")
        stars = self.__data.stars
        center = pg.analysis.shrinking_sphere(stars, center=[stars.boxsize / 2] * 3, R=stars.boxsize)
        pg.Translation(-center).apply(self.__data, total=True)
        self.__center = center

        # Rvir_property = self.__snap_config.Rvir_property
        # Rvir = self.dm_halo.props[Rvir_property]
        Rgx = float(self.Rvir) * self.gx_radius
        mask = pg.BallMask(str(Rgx) + ' kpc h_0**-1', center=[0, 0 , 0])
        snapshot_gx = self.__data[mask]

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("searching for galaxies - complete")
            print("*********************************************")

        self.__galaxy = snapshot_gx
        self.__galaxy_all = self.__galaxy
        return


    def __gx_orientate(self):
        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("*********************************************")
            print("orientate galaxy...")

        pg.analysis.orientate_at(self.__galaxy_all, 'L', total=True)

        if pg.environment.verbose >= pg.environment.VERBOSE_TACITURN:
            print("orientate complete")
            print("*********************************************")
        return


    def print_results(self):
        print(" ")
        print("**********************************************************")
        print(("Summary of results for snapfile ", self.name))
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
            if v != value:
                self[name] = value
                self.__modified = True
        else:
            v = None
            self[name] = value
            self.__modified = True
        return v        # returns old value

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
                                value = float(value) if value != 'None' else None
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
