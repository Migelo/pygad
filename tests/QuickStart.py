# %% [markdown]
# # Quickstart to `pygad`

# %% [markdown]
# ## Content:
# 
# 1. loading the snapshot and accessing data
#     - 1.1 load a `Snap`
#     - 1.2 access blocks and other data
#     - 1.3 blocks have units
#     - 1.4 sub-snapshots
#     - 1.5 dervied blocks
#     - 1.6 more on blocks and `UnitArrs`
# 2. analysis and plotting
#     - 2.1 a first look
#     - 2.2 finding halos & preparing a zoom
#     - 2.3 plotting maps and other stuff
#     - 2.4 binning
#     - 2.5 quantitative analysis

# %% [markdown]
# ## 1. loading the snapshot and accessing data

# %% [markdown]
# Let us first prepare the environment (not neccesary, but nice) and import `pygad`:

# %%
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pygad as pg

warnings.filterwarnings("ignore")

# %% [markdown]
# ### 1.1 load a `Snap`

# %% [markdown]
# Loading snapshots is easy! `pygad` supports format 1, format 2 (utilizing the info block), and HDF5:

# %%
s = pg.Snapshot("./pygad/snaps/snap_M1196_4x_470", load_double_prec=True)
print(s)

# %% [markdown]
# We force to load all blocks in double precision by passing `load_double_prec=True`. The format is automatically detected as well as its endianess, whether it is a cosmological run, and iformation of the blocks (you might need to do some adjustments in the config files, though).

# %%
s.cosmological

# %% [markdown]
# ### 1.2 access blocks and other data

# %% [markdown]
# Blocks are only loaded when needed. They can be accessed by by their lower-case names as "items": `s['pos']`. The data is a `SimArr`, a class derived from `UnitArr` which in turn is derived from `np.ndarray` and enhanced with units (which have only a small impact on performance). They can easily be converted to other units (many are predefined; one can, however also define some oneself. **Attention:** `'h'` is for hours and `'h_0'` is for the Hubble parameter $h_0$ in $H_0 = 100 \ h_0 \ \mathrm{km/s} \ / \ \mathrm{Mpc}$):

# %%
s["pos"]

# %% [markdown]
# Some general properties of the snapshot are `s.boxsize`, `s.parts` (#particles per particle type), `s.redshift`, `s.scale_parameter`, and `s.time`. The last one is the time parameter of the header and, hence, identical to `s.scale_parameter` for cosmological runs.

# %%
s.parts

# %%
s.boxsize

# %%
np.abs(s.scale_factor - s.time) < 1e-15

# %% [markdown]
# There is also `s.cosmic_time`, which gives the time since the Big Bang.

# %%
s.cosmic_time()

# %% [markdown]
# More header information is accessible via `s.properties`:

# %%
s.properties

# %% [markdown]
# Another usefule attribute is the cosmology of the snapshot as read from the header, represented as a class for Friedmann-Lemaître-Robertson-Walker cosmologies (other cosmologies are available in `pygad.cosmology`):

# %%
s.cosmology

# %% [markdown]
# This class provides a lot of useful functions. For instance, you can calculate the physical size of an object at redshift 1.2 that appears under the apparent angle of 0.5":

# %%
s.cosmology.angle_to_length("0.5 arcsec", z=1.2)

# %% [markdown]
# Please, explore the powers of this class with the Python built-in functions `dir` (or iPython's tab completion) and `help` – every function and class in `pygad` is documented!

# %% [markdown]
# ### 1.3 blocks have units

# %% [markdown]
# Back to the acutal data, the blocks of the snapshot. As said, they are instances of `SimArr`, derived from `UnitArr`. The relevant difference for you as a user is, that the `SimArr`s also have a refernce to the snapshot and by its cosmology know the values of the scale parameter `a` (for converting between comoving and porper lengths), the redshift `z`, and the Hubble parameter `h_0`. Like `UnitArr`s they have units and can esily be converted:

# %%
print("inplace conversion from", s["pos"].units, s["pos"].convert_to("kpc"))
# a UnitArr would need subsitutions for 'a' and 'h_0' - see below
print("to", s["pos"].units)
print('the UnitArr as a result of conversion without chaning s["pos"]:')
s["pos"].in_units_of("AU")

# %% [markdown]
# At loading the Gadget units as defined in the config files are assumed. Typically these are comoving units with factors of `h_0`. You can request to convert all blocks (also those that get loaded later) to physical units, that is no comoving units and `h_0` factored out:

# %%
s.to_physical_units()

# %% [markdown]
# When doing calculations with the blocks, they are converted into normal `UnitArr`s. It is kept track of the units as with all `UnitArr`s:

# %%
np.median(s["mass"] * np.sum(s["vel"] ** 2, axis=-1)).in_units_of("erg")

# %% [markdown]
# ### 1.4 sub-snapshots

# %% [markdown]
# A snapshot consists of different particle types that are grouped into different particle families, such as gas, baryons, and dark matter (by default abbreviated to `dm`). These families are defined in the config files and can be accessed via attribute names:

# %%
s.gas

# %% [markdown]
# This is a sub-snapshot, a snapshot that is just a subset of its root. It has all the properties of the root and can be used like a normal snapshot. Blocks are masked appropiately such that `s.baryons['pos']` are the postions of the baryons only:

# %%
s.baryons["pos"]

# %%
len(s.baryons["pos"]) == len(s.baryons)

# %% [markdown]
# Sub-snapshots can also be created by slices and masks analoguous to numpy arrays:

# %%
s[10:200:3]

# %%
mask = np.zeros(len(s), dtype=bool)
mask[:10] = True
mask[::1000] = True
s[mask]

# %% [markdown]
# Moreover, `pygad` has snapshot mask classes (defined in `pygad.snapshot.masks`) that are specified for common masking task as cutting out a spherical region:

# %%
ball = s[pg.BallMask("1 Mpc", center=pg.UnitArr([48.072, 49.346, 46.075], "Mpc"))]
ball

# %% [markdown]
# The different masks can be combined and all the nasty slicing and masking of the blocks is done for you:

# %%
ball[::3].baryons["pos"]

# %%
ball[::3].baryons.parts

# %% [markdown]
# **Note**: Not all blocks are for all particles and you will get an exception, if you try to access those from the root snapshot ('rho' is for gas only):

# %%
try:
    s["rho"]
except KeyError as e:
    print("KeyError:", e)

# %% [markdown]
# A list of the blocks available for a certain (sub-)snapshot is returned by the method `available_blocks`.

# %%
"rho" in s.available_blocks(), "rho" in s.gas.available_blocks()

# %% [markdown]
# Sometimes one want to have a more complex mask than just a ball or box (e.g. a certain region in the phase diagram) and apply it to different snapshots. For such purpose pygad provides `ExprMask`, which takes a string that is Python code using the blocks (here some functions are used that are explained later):

# %%
pg.environment.verbose = pg.environment.VERBOSE_QUIET

# some pressure threshold:
highP = pg.ExprMask("temp * rho * R / UnitScalar('1 g/mol') > '1e-18 N/m**2'")
# you can also do even more complex masking by combing masks:
lowPhighZ = (~highP) & pg.ExprMask("metallicity > 0.1*solar.Z()")
# or directly (note the parenthesis!):
# highPhighZ = pg.ExprMask("(temp * rho * R / UnitScalar('1 g/mol') "
#                         "> '1e-18 N/m**2') & (metallicity > 0.1*solar.Z())")

# use the high pressure mask
mask = highP

fig, ax = plt.subplots(figsize=(4, 3))
pg.plotting.phase_diagram(
    s.gas[mask],
    rho_units="g/cm**3",
    extent=[[-31, -22], [2.3, 7]],
    colors="metallicity/solar.Z()",
    colors_av="mass",
    clim=[3e-2, 0.3e1],
    clogscale=True,
    showcbar=True,
    cbartitle=r"[Z]",
    ax=ax,
)

print("masked fraction for ...")
print(
    "   all gas:      %4.1f%%" % (1e2 * s.gas[mask]["mass"].sum() / s.gas["mass"].sum())
)
print(
    "   gas in ball:  %4.1f%%"
    % (1e2 * ball.gas[mask]["mass"].sum() / ball.gas["mass"].sum())
)
fig.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")

# Move entire simulation (for easier plotting); explained in more detail later
ball_center = pg.UnitArr([48.072, 49.346, 46.075], "Mpc")
pg.Translation(-ball_center).apply(s)

for mask_idx, mask in enumerate([~mask, mask]):
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    args = dict(extent="2 Mpc", vlim=[3e3, 1e6], clim=[1e4, 1e6], Npx=50)  # 100)
    pg.plotting.image(s.gas[mask], xaxis=0, yaxis=1, ax=ax[0, 0], **args)
    pg.plotting.image(s.gas[mask], xaxis=0, yaxis=2, ax=ax[1, 0], **args)
    pg.plotting.image(s.gas[mask], xaxis=2, yaxis=1, ax=ax[0, 1], **args)
    ax = ax[1, 1].set_axis_off()
    fig.tight_layout()
    fig.savefig("gas_images_{}.png".format(mask_idx), dpi=300, bbox_inches="tight")

pg.Translation(ball_center).apply(s)
pg.environment.verbose = pg.environment.VERBOSE_NORMAL

# %% [markdown]
# ### 1.5 dervied blocks

# %% [markdown]
# There are also some blocks additional to the blocks from the snapshot file. These are so-called "derived blocks" that are calculated from other blocks (as defined in the `derived.cfg`) and automatically updated, if the underlying data changes (with lazy evaluation, i.e. they are only (re-)calculated when needed).

# %% [markdown]
# A example for such an derived block are the particle radii (distance form origin):

# %%
s["r"]

# %%
print(s["r"][0])
# if no units are given, the ones of the UnitArr are taken,
# meaning we substract [1e4,1e4,1e4]*s.pos.units
s["pos"] -= [1e4, 1e4, 1e4]
print(s["r"][0])
s["pos"] += [1e4, 1e4, 1e4]
print(s["r"][0])

# %% [markdown]
# The caching of the derived blocks, of course, needs some memory. It, however, can be turned off. Some blocks with time-consuming calculation can still be cached though (list is specified in the config file `derived.cfg`):

# %%
s.cache_derived = False
s.always_cached_derived

# %%
del s["r"]
s["r"]
s["r"]
s.cache_derived = True

# %% [markdown]
# ### 1.6 more on blocks and `UnitArr`s

# %% [markdown]
# Blocks that get deleted can always be re-loaded / re-derived by simply accessing them as shown above. The only exception are custom blocks you added yourself:

# %%
s["custom"] = np.ones(len(s))
s["custom"]

# %%
del s["custom"]
try:
    s["custom"]
except KeyError as e:
    print("KeyError:", e)

# %% [markdown]
# You can get new blocks calculated from existing ones with the method `get` (not that this one makes much sense...):

# %%
s.gas.get("log10(temp) * mass")

# %% [markdown]
# When calculating with blocks, they get converted if necessary:

# %%
pg.UnitArr([1, 2, 3], "km/h") + pg.UnitArr([1.2, 1.3, 1.4], "m/s")

# %% [markdown]
# If units contain cosmological parameter such as the scale parameter in comoving units, you need to specify them by `subs`(titutions). This can either be a `dict` or a snapshot instance:

# %%
pg.UnitArr([1.23, 4.56], "ckpc/h_0").in_units_of("kpc", subs={"a": 0.5, "h_0": 0.7})

# %%
pg.UnitArr([1.23, 4.56], "ckpc/h_0").in_units_of("kpc", subs=s)

# %% [markdown]
# You can ensure to have a `UnitArr` rather than a simple `list` of a `np.ndarray` with `UnitQty` and at the same time ensure certain units and `dtype`:

# %%
pg.UnitQty([1, 2, 3], "kpc")

# %%
pg.UnitQty(pg.UnitArr([1, 2, 3], "kpc"), "Mpc", dtype=float)

# %% [markdown]
# Similar is `UnitScalar` that additionally ensure that the result is a scalar:

# %%
pg.UnitScalar("1.2 kpc/Mpc")

# %% [markdown]
# And these can be converted into floats with proper conversion into dimensionless units, if possible:

# %%
float(pg.UnitScalar("1.2 kpc/Mpc"))

# %% [markdown]
# **caveat:** if the units cannot be converted to dimensionless ones, they are simply cut off. Hence, it is

# %%
float(pg.UnitScalar("1.2 Msol kpc/Mpc"))

# %% [markdown]
# ## 2. analysis and plotting

# %% [markdown]
# Let us first make `pygad` less talky:

# %%
pg.environment.verbose = pg.environment.VERBOSE_TACITURN

# %% [markdown]
# ## 2.1 a first look

# %% [markdown]
# There is a very general plotting function in `pygad`. If it is not specified what to plot, it automatically chooses these plotting quantities from the type of the sub-snapshot passed.

# %%
f, ax = plt.subplots()
pg.plotting.image(s.gas, ax=ax)
f.savefig("gas_image.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Here, the luminance of the image scales with the logarithm of the column density and the color is (density-weighted) temperature as indicated by the colorbar. The extent of the plot is chosen such that most of the particle are within the plotted region. We are here apparently dealing with a zoomed-in simulation.
# 
# The smoothing of the SPH particles is always properly taken into account and the mapping is integral conserving (which is not true for all the plotting routines out there!).
# 
# More on plotting follows below. Still, I want to emphasise again that every function (and class) in `pygad` is documented! So use `help(pygad.plotting.image)`.

# %% [markdown]
# ## 2.2 finding halos & preparing a zoom

# %% [markdown]
# In galaxy formation simulations we typically what to identify halos and galaxies and then do analysis on them. If we know all particle IDs of a halo or galaxy (e.g. taken from a halo finder), we could just use `pygad.IDMask` for masking out the structure of interest as a sub-snapshot and then use the machinery of `pygad` (or self-written tools building on `pygad`).

# %% [markdown]
# ### 2.2.1 shrinking sphere

# %% [markdown]
# However, `pygad` offers a few tools for identifying halos and galaxies. One simple approach in a zoomed-in simulation (as the one used here) is the shrinking sphere method:

# %%
center = pg.analysis.shrinking_sphere(s.stars, center=[s.boxsize / 2] * 3, R=s.boxsize)

# %% [markdown]
# ... and then just do a translation of the entire snapshot:

# %%
pg.Translation(-center).apply(s)
print(center)

# %% [markdown]
# This way of doing the translation has the advantage that the snapshot stores the action and applies the translation to the positions each time they are reloaded:

# %%
pg.environment.verbose = pg.environment.VERBOSE_TALKY
del s["pos"]
s["pos"]
pg.environment.verbose = pg.environment.VERBOSE_TACITURN

# %% [markdown]
# There are some more such transformations (as a rotation) in `pygad.transformation`.

# %% [markdown]
# ### 2.2.2 `pygad`'s FoF finder

# %% [markdown]
# `pygad` also offers a friends-of-friends (FoF) finder for identifiying halos. In fact, it is more than just a simple FoF finder, since one can also specify a maximum velocity difference for particles becoming "friends".

# %%
# by default the velocity criterion is not used
FoF, N_FoF = pg.analysis.find_FoF_groups(s.dm, l="6 ckpc")

# %% [markdown]
# Note how the most massive dark matter halo, indeed is roughly at the center found by the shrinking sphere (though it was on stars not dark matter).
# 
# `FoF` is a array with halo indices sorted by mass (`pygad.analysis.NO_FOF_GROUP_ID` for particles without a halo). So the group with ID 0 is the most massive one. That could now be used for masking:

# %%
halo = s.dm[FoF == 0]
halo["mass"].sum()

# %% [markdown]
# More convenient is, however, can be using `pygad.analysis.generate_FoF_catalogue` which returns a list of `Halo` classes. These have the advantage to have some basic properties, such as total mass, center of mass, and virial radius, and they can be used to mask the halos by IDs so that you don't have to mask excalty the sub-snapshot you passed to `pygad.analysis.find_FoF_groups`. The downside is that calculating the halo properties takes some time, which, however, can be reduced by limiting the catalogue to only the most massive ones by setting `max_halos`. Additionally, one can excluding some for which a function of halo and snapshot passed as `exclude` returns `True`.

# %%
halos = pg.analysis.generate_FoF_catalogue(
    s.dm, FoF=FoF, max_halos=5, exlude=lambda h, s: h.lowres_mass / h.mass > 0.01
)

# %% [markdown]
# Passing `FoF` can be omitted, then `generate_FoF_catalogue` calls `find_FoF_groups` itself.

# %%
halo = halos[0]
halo.props

# %% [markdown]
# The halo instances can be used to mask the snapshot to the sub-snapshots fo the halos:

# %%
h = s[halo]
h

# %% [markdown]
# ... or using its properties to mask everying within $R_{200}$:

# %%
h = s[pg.BallMask(halo.R200_ssc, center=halo.ssc)]
print(h.parts)
print(h.dm["mass"].sum())

# %% [markdown]
# The velocity criterion can be handy for definig the galaxy:

# %%
pg.environment.verbose = pg.environment.VERBOSE_NORMAL
galaxies = pg.analysis.generate_FoF_catalogue(
    s.baryons, l="6 ckpc", dvmax="100 km/s", max_halos=5
)
pg.environment.verbose = pg.environment.VERBOSE_TACITURN

# %%
s[galaxies[0]]

# %% [markdown]
# ## 2.3 plotting maps and more

# %% [markdown]
# ### 2.3.1 spatial maps

# %% [markdown]
# The main plotting function `pygad.plotting.image` automatically chooses colorbar and extent depending on what particle families are plotted. Alongside are the plotting quantities chosen. Generally this is column density. If only gas or stars are plotted the image is composed of two chanels, namely luminance for one quantity and color for another one. In case of the gas, the luminance is column denisty and color is density-weighted temperature. For stars only, the luminance is V-band luminosity (as from SSP models) and the color is V-band weighted age.

# %%
fig, ax = plt.subplots(2, 2, figsize=(9, 9))
pg.plotting.image(s, extent="250 kpc", ax=ax[0, 0])
pg.plotting.image(s.dm, extent="250 kpc", ax=ax[0, 1])
pg.plotting.image(s.gas, extent="250 kpc", ax=ax[1, 0])
pg.plotting.image(s.stars, extent="50 kpc", ax=ax[1, 1], Npx=150)
fig.tight_layout()
fig.savefig("snap_all_dm_gas_stars.jpg", dpi=300)

# %% [markdown]
# These are, however, only the defaults for `image`! The parameters can be adjusted by giving arguments to `image`.
# 
# Not that the colormap will be normed in luminosity, if both chanels, color (or more precisely) hue and luminosity, are used. It makes sense to choose a bright one. Note that not all colors can be equally bright (to the human eye). Pure blue ([0,0,1] in RGB space), for instance, is much darker than pure yellow ([1,1,0] in RGB space).
# 
# So for plotting with `image`:
# 
# * When combing the luminance and color information, use bright colormaps. I can recommend the newly defined `'Bright'` (similar to `'jet'` or `'rainbow'`, but brighter; often the default in `pygad` as for gas temperature), the also in `pygad` defined `'NoBlue[_r]'` or the standard `'PuOr_r'`, `'coolwarm'`, and `'spring'`. For some Python distributions there is also `'plasma'` and `'viridis'`.
# * If you want to have such a combination of luminance channel and color chanel, specify both, `qty` and `colors`. If you only pass a `qty` to plot, it is plotted using the given colormap and the luminance is not used.
# * Both, `qty` and `colors` can be averaged by some other quantity given by `av` or `colors_av`, respectively.
# * You can define the axis / orientation by `xaxis` and `yaxis`.
# * By default a surface density is plotted. This means, if you set `qty='mass'`, actually the column density is plotted. To plot the quantity summed along the line of sight (or averaged, if `av` is given) set `surface_dens=False`. * You can set to plot in log-scale. If you set something like `qty='log10(temp)'`, note that already the given quantity is logarithmic. Don't forget to adjust the limits accordingly!
# * Extent and resolution of the map can be specified independently for the two directions.

# %% [markdown]
# **Example:** a more complicated plotting example including some simple masking:

# %%
args = dict(
    extent=pg.UnitArr([[-1.5, 3.0], [-1.0, 2.0]], "Mpc"),
    Npx=[4 * 90, 4 * 60],
    qty="mass",
    av=None,
    field=False,
    surface_dens=True,
    units="Msol/kpc**2",
    vlim=[2e3, 2e6],
    colors="temp",
    colors_av="mass",
    clogscale=True,
    cbartitle=r"$\log_{10}(T\,[\mathrm{K}])$",
    clim=[10.0**3.5, 10.0**5.8],
    cmap="plasma",
    desat=0.33,
    xaxis=1,
    yaxis=2,
    scaleunits="Mpc",
    fontcolor="w",
    fontsize=17,
)

# simple boolean masks
cold = s.gas["temp"] < "2e4 K"
warm = (s.gas["temp"] < "1e5 K") & ~cold
hot = s.gas["temp"] >= "1e5 K"

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig, ax0, im0, cbar0 = pg.plotting.image(s.gas[cold], ax=ax[0, 0], **args)
fig, ax1, im1, cbar1 = pg.plotting.image(s.gas[warm], ax=ax[0, 1], **args)
fig, ax2, im2, cbar2 = pg.plotting.image(s.gas[hot], ax=ax[1, 0], **args)
fig, ax2, im2, cbar2 = pg.plotting.image(s.gas, ax=ax[1, 1], **args)
fig.tight_layout()
fig.savefig("snap_gas_temps.jpg", dpi=300, bbox_inches="tight")

del cold, warm, hot

# %% [markdown]
# Explain `field` and `surface_density`:

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pg.plotting.image(
    s.gas,
    extent="200 kpc",
    Npx=100,
    qty="mass",
    field=False,
    surface_dens=True,
    ax=axs[0],
)
pg.plotting.image(s.gas, extent="200 kpc", Npx=100, qty="rho", ax=axs[1])
fig.savefig("snap_gas_field_surface_density.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# As already mentioned, SPH smoothing is always correctly taken into account. It is also ensured that no particles "fall through the grid" and the resulting map is integral conserving. That is, if you plot mass (as a surface density) the integral over the map is the total mass in the map. A naive binning technique where the SPH density is evaluated at each pixel center would not give such a map!

# %% [markdown]
# It follows an example plotting stellar quantities. As those are point particles, one ends up with a very grainy map, if the resolution is high. They, however, can also be smoothed by a softening length when passed to the plotting function by the argument `softening`.

# %%
fig, ax, im, cbar = pg.plotting.image(
    s.stars,
    extent="55 kpc",
    Npx=200,
    qty="metallicity/solar.Z()",
    av="mass",
    surface_dens=False,
    vlim=[10**-0.25, 10**0.15],
    softening=pg.UnitArr([0, 2, 5, 10, 1, 1], "kpc"),
    fontcolor="w",
    fontsize=22,
    cbartitle="[Z]",
)
cbar.set_ticks(
    [
        -0.2,
        -0.1,
        0,
        0.1,
    ]
)

fig.savefig("snap_stars_metallicity.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### 2.3.2 on the color-coding

# %% [markdown]
# Tell about why 'jet' is bad and shortly about color-spaces (rgb vs. hsv and human perception).

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
pg.plotting.image(
    s.gas,
    extent="200 kpc",
    Npx=100,
    qty="rho",
    vlim=[10**5.5, 10**7.5],
    cmap="gray",
    cbartitle="col. dens. > value",
    ax=axs[0, 0],
)
pg.plotting.image(
    s.gas,
    extent="200 kpc",
    Npx=100,
    qty="log10(temp)",
    av="rho",
    surface_dens=False,
    units=1,
    vlim=[3, 6],
    logscale=False,
    cmap="isolum",
    cbartitle="temp. > hue",
    fontcolor="k",
    ax=axs[0, 1],
)
pg.plotting.image(
    s.gas,
    extent="200 kpc",
    Npx=100,
    qty="rho",
    vlim=[10**5.5, 10**7.5],
    colors="log10(temp)",
    colors_av="rho",
    csurf_dens=False,
    cunits=1,
    clim=[3, 6],
    cmap="isolum",
    cbartitle="combined",
    ax=axs[1, 0],
)
axs[1, 1].axis("off")

fig.savefig("jetbad.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### 2.3.3 plotting slices (of finite thickness)

# %% [markdown]
# SPH binnning routines in pygad ensure that nothing "falls through the grid" by normalising the contribution of each particle by the discrete integral of its kernel over the grid. Whenever this is happens to be zero (the particle falls through the grid entirely), the particle is fully added to the closes cell. This can, of course, only be applied to particles that are entirely within in the grid.

# %% [markdown]
# Here we create a 1 kpc thick slice, padding with a few additional voxels/cells to make use of the normation described in order to not let anything fall through the grid.

# %%
m = pg.binning.SPH_to_3Dgrid(
    s.gas, "rho", extent=pg.UnitArr([100, 100, 11], "kpc"), Npx=[100, 100, 11]
)
fig, ax = plt.subplots(1)
pg.plotting.plot_map(m[:, :, 5], vlim=[1e3, 1e7])
fig.savefig("to3Dgrid.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# Let us zoom in a bit and plot linearly to see the differences between the normation in pygad and what it looks like without:

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

pltargs = dict(units="1e-24 g/cm**3", vlim=[0, 1.1], logscale=False)

for ax, normed in zip(axs, [True, False]):
    m = pg.binning.SPH_to_3Dgrid(
        s.gas,
        "rho",
        extent=pg.UnitArr([50, 50, 11], "kpc"),
        Npx=[50, 50, 11],
        normed=normed,
    )
    pg.plotting.plot_map(m[:, :, 5], ax=ax, **pltargs)
    ax.set_title(("" if normed else "not ") + "normed", fontsize=14)
fig.savefig("to3Dgrid_normed.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# density weigthed temperature (as the simulation is a multi-phase one and one would otherwise overshadow any small temperature as they vary over orders of magnitude):

# %%
m = pg.binning.SPH_to_3Dgrid(
    s.gas, "rho*temp", extent=pg.UnitArr([100, 100, 5.5], "kpc"), Npx=[200, 200, 11]
)
m /= pg.binning.SPH_to_3Dgrid(
    s.gas, "rho", extent=pg.UnitArr([100, 100, 5.5], "kpc"), Npx=[200, 200, 11]
)
fig, ax = plt.subplots(1)
pg.plotting.plot_map(m[:, :, 5], ax=ax)
fig.savefig("to3Dgrid_rho_temp.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# ### 2.3.4 plot vector fields

# %% [markdown]
# For overplotting maps with vector fields, you can use `pg.plotting.vec_field`, which uses either `plt.quiver` or `plt.streamplot` (depending on the argument `streamplot`) to plot the vecot field. Please, also read the documentation of `matplotlib` on these functions!

# %%
fig, ax = plt.subplots()
pg.plotting.image(s.gas, extent="200 kpc", Npx=100, ax=ax)
pg.plotting.vec_field(
    s.gas,
    "vel",
    av="mass",
    color="w",
    extent="200 kpc",
    Npx=25,
    ax=ax,
    angles="xy",
    units="xy",
    scale_units="xy",
    scale=10.0,  # passed to plt.quiver
)
fig.savefig("vec_field.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### 2.3.5 maps in other spaces

# %% [markdown]
# Although a phase diagram is not a map in the sense that the coordinates of the plot are spatial postions it is a map. Te procedure of plotting is very similar. Here is a phase diagram colorcoded by mass-weighted metallicity (`u` is the atomic mass unit):

# %%
fig, ax = plt.subplots()
pg.plotting.phase_diagram(
    s.gas,
    rho_units="g/cm**3",
    extent=[[-34, -22], [2.2, 7]],
    colors="metallicity",
    colors_av="mass",
    clogscale=True,
    clim=[1e-4, 5e-2],
    fontcolor="w",
    fontsize=18,
    showcbar=True,
    threshold_col="w",
    T_threshold="8e4 K",
    rho_threshold="1 u/cm**3",
    ax=ax
)
fig.savefig("phase_diagram_metallicity_T.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# A more general routine to produce such a 2D-histogram like plot is `pygad.plotting.scatter_map`. It basically is a genearl plotting function for 2D histograms:

# %%
fig, ax = plt.subplots(figsize=(5, 5))
pg.plotting.scatter_map(
    np.random.normal(size=int(1e5)),
    np.random.normal(size=int(1e5)),
    extent=[[-3, 3], [-3, 3]],
    fontcolor="w",
    bins=50,
    ax=ax,
)
fig.savefig("scatter_map.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# ... but with added functionality, especially for snapshots:

# %%
fig, ax = plt.subplots()
pg.plotting.scatter_map(
    "log10(metallicity+1e-15)",
    "log10(temp)",
    s.gas,
    bins=150,
    logscale=True,
    extent=[[-4, -1], [2.2, 6]],
    colors="log10(rho)",
    colors_av=np.ones(len(s.gas)),
    cbartitle=r"$\log_{10}(\rho\,[%s])$" % s.gas["rho"].units.latex(),
    clim=[0, 9],
    fontcolor="lightgray",
    fontsize=18,
)
fig.savefig("scatter_map_metallicty_T.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### 2.3.6 plotting profiles

# %% [markdown]
# `pygad` also provides plotting routines for profiles of arbitrary quantities in either 3D or projected along one coordinate axis. As an example, I plot (face-on) surface densities of stars and gas.
# 
# *Note* that in the version of `pygad` used here, the SPH smoothing is not taken into account.

# %%
pg.__version__

# %%
fig, ax = pg.plotting.profile(
    s.stars,
    Rmax="50 kpc",
    qty="mass",
    proj=2,
    units="Msol/pc**2",
    linewidth=5,
    color="orange",
    label="stars",
)
pg.plotting.profile(
    s.gas,
    Rmax="50 kpc",
    qty="mass",
    proj=2,
    units="Msol/pc**2",
    linewidth=5,
    color="blue",
    ax=ax,
    label="gas",
    labelsize=18,
)
ax.set_ylim([1e-1, 1e3])
ax.set_xlim([0, 50])
ax.legend(fontsize=18)
fig.savefig("profile.jpg", dpi=300, bbox_inches="tight")

# %% [markdown]
# # TODO from 3D grids...

# %% [markdown]
# ... also implement in `pygad`

# %%
m = pg.binning.SPH_to_3Dgrid(s.gas, "rho", extent="100 kpc", Npx=200)

# %% [markdown]
# ## 2.4 binning

# %% [markdown]
# The plotting routine `image` heavily depens on two fundamental binning routines for SPH properties. One bins onto a 3D grid (`pygad.binning.SPH_to_3Dgrid`) and the other one onto a projected 2D map (`pygad.binning.SPH_to_2Dgrid`). The latter is almost a factor of 100 faster than first binning onto a 3D grid and then project and needs more than a factor of 100 less memory for typical situations where the 3D grid would have several 100 pixels in depth.

# %% [markdown]
# The 3D version works similar to the 2D one. Therefore, I demonstrate only the use of the projected version here:

# %%
map2D = pg.binning.SPH_to_2Dgrid(s.gas, "rho", extent="100 kpc", Npx=100)
map2D.res()

# %%
map2D

# %%
map2D.grid

# %% [markdown]
# Also the quantitative profile as intermediate step for plotting profiles can be accessed directly. There is `radially_binned` and `profile_dens` in `pygad.analysis`.

# %% [markdown]
# ## 2.5 quantitative analysis

# %% [markdown]
# ### 2.5.1 defining discs

# %% [markdown]
# Colorful maps are neat and can help to understand the simulations, but in the end we are physicists and want quantitative results. Only a few of the analysis functions presented so far are of such type. There is, of course, more to that in `pygad`. Let us start by orientating our simulation at the "reduced inertia tensor" (Raini & Steinmetz, 2005) of the galaxy after centering it in coordinate and velocity space:

# %%
g = s.baryons[pg.BallMask(0.15 * halo.R200_com)]

center = pg.analysis.center_of_mass(g)
pg.Translation(-center).apply(s)

vel_center = pg.analysis.mass_weighted_mean(g[pg.BallMask("1 kpc")], "vel")
s["vel"] -= vel_center

# using `transformation.Rotation` in the backend
pg.analysis.orientate_at(g, "L", total=True)

# %% [markdown]
# Giving `total=True` lets the entire snapshot orientate, not just the passed sub-snapshot. Once orientated, we can decompose the galaxy into its disc and the rest of the galaxy using a predefined mask of `pygad`:

# %%
disc_mask = pg.DiscMask(jzjc_min=0.85, rmax=None, zmax="3 kpc")
disc = g[disc_mask]
nondisc = g[~disc_mask]  # all masks in pygad can be inverted with ~

# %% [markdown]
# The main criterion to differentiate between the disc and the rest in `DiscMask` is the parameter `jzjc`, the ratio between the z-component of the angular momentum $j_z$ of an particle and the angular momentum $j_c$ of an particle with the same energy but on a circular orbit:

# %%
fig, ax = plt.subplots()
ax.hist(
    [nondisc["jzjc"], disc["jzjc"]],
    bins=50,
    range=(-1, 1),
    histtype="stepfilled",
    stacked=True,
    log=True,
)
ax.set_xlim([-1, 1])
ax.set_ylim([3e2, 3e4])
ax.set_ylabel("particles", fontsize=18)
ax.set_xlabel("$j_z / j_c$", fontsize=20);
fig.savefig("jzjc_hist.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# A plot to demonstrate the power of the decomposition:

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
pg.plotting.image(disc.stars, extent="45 kpc", Npx=210, ax=ax[0], scaleind=None)
pg.plotting.image(nondisc.stars, extent="45 kpc", Npx=210, ax=ax[1])
fig.tight_layout()
fig.savefig("disc_nondisc_face.jpg", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(8, 1.8))
pg.plotting.image(
    disc.stars,
    extent=pg.UnitArr([45, 15], "kpc"),
    Npx=[210, 70],
    yaxis=2,
    ax=ax[0],
    showcbar=False,
    scaleind=None,
)
pg.plotting.image(
    nondisc.stars,
    extent=pg.UnitArr([45, 15], "kpc"),
    Npx=[210, 70],
    yaxis=2,
    ax=ax[1],
    showcbar=False,
    scaleind=None,
)
fig.tight_layout()
fig.savefig("disc_nondisc_edge.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# We now can calculate kinematical disc-to-total mass and luminosity ratios of the stars:

# %%
float(disc.stars["mass"].sum() / g.stars["mass"].sum())

# %%
float(disc.stars["lum"].sum() / g.stars["lum"].sum())

# %%
float(disc.stars["lum_v"].sum() / g.stars["lum_v"].sum())

# %% [markdown]
# Using numpy we can calculate median ages of the disc and the rest. As already seen from the maps, the disc is much younger:

# %%
print("disc:", np.median(disc.stars["age"]))
print("rest:", np.median(nondisc.stars["age"]))

# %% [markdown]
# Or as a diagram:

# %%
fig, ax = plt.subplots()
ax.hist(
    [nondisc.stars["age"], disc.stars["age"], g.stars["age"]],
    label=("not in disc", "disc", "total"),
    color=("r", "b", "k"),
    bins=30,
    range=(0, s.cosmology.universe_age()),
    histtype="step",
    stacked=False,
    linewidth=5,
)
ax.set_xlim([0, s.cosmology.universe_age()])
ax.set_ylim([0, 6.5e3])
ax.set_ylabel("stellar particles", fontsize=16)
ax.set_xlabel("stellar age $[%s]$" % disc.stars["age"].units.latex(), fontsize=16)
ax.legend(loc="upper left", fontsize=14);
fig.savefig("age_hist.jpg", dpi=300, bbox_inches="tight")
# %% [markdown]
# ### 2.5.2 half-mass and half-light radii

# %% [markdown]
# Often half-mass and/or effective radii are of interest. `pygad` can calculate them in either 3D or projections along one of the three coordinate axis:

# %%
pg.analysis.half_mass_radius(g.stars, proj=None)  # 3D half-mass radius

# %%
pg.analysis.half_mass_radius(g.stars, proj=2)

# %% [markdown]
# The effective radius can be calculate in any band using the single stellar population (SSP) models of Bruzual & Charlot (however, no gas and dust extinction is included):

# %%
pg.analysis.eff_radius(g.stars, band="V", proj=2)

# %% [markdown]
# And half-quantity radii can also be calculated of arbitrary other quantities:

# %%
pg.analysis.half_qty_radius(g.stars, "metals", proj=2)

# %% [markdown]
# ### 2.5.3 virial radius and mass

# %% [markdown]
# Given the center of some halo, `virial_info` calculates the "virial" radius and "virial" mass in terms of spherical overdensity thresholds you can define. By default it is $R_{200}$ and $M_{200}$.

# %%
pg.analysis.virial_info(s)

# %%
pg.analysis.virial_info(s, odens=500, center=[0] * 3)

# %% [markdown]
# ### 2.5.4 ionisation states & mock absoprtion spectra

# %% [markdown]
# I also want to draw some attention to the sub-module `pygad.analysis.absorption_spectra` and the derived blocks for ion masses.

# %% [markdown]
# Let's define some line of sights (l.o.s.) along given coordinate axis and a line to look at. Line properties for some chosen ones are predefined in `pygad.analysis.absorption_spectra.lines`.

# %%
loss = pg.UnitArr(
    [
        [50.0, 100.0],
        [-180.0, -100.0],
        [170.0, -150.0],
        [-110.0, 10.0],
        [-45.0, 5.0],
        [48.0, -20.0],
        [250.0, -200.0],
    ],
    "kpc",
)
xaxis, yaxis, zaxis = 0, 2, 1
line_name = "Lyman_alpha"
line = pg.analysis.absorption_spectra.lines[line_name]
for key, value in line.items():
    print("%-10s %s" % (key + ":", value))

# %%
print(
    len(pg.analysis.absorption_spectra.lines),
    ", ".join(list(pg.analysis.absorption_spectra.lines.keys())),
)

# %% [markdown]
# The line properties are influenced by some gas properties: most and formost the column density, of course, but also the l.o.s. velocity and its dispersion (or actually the precise distribution) and the (ion) temperature. So let's plot some maps with the line positions indicated:

# %%
fig, axs = plt.subplots(2, 2, figsize=(7, 7))
d = 300.0
pltargs = dict(
    extent=pg.UnitArr([[-d, d], [-d, d]], "kpc"),
    xaxis=xaxis,
    yaxis=yaxis,
    Npx=100,
)

units = pg.UnitArr(line["atomwt"]) * pg.Unit("cm**-2")
units = float(units) * units.units

pg.plotting.image(
    s.gas,
    line["ion"],
    field=False,
    surface_dens=True,
    units=units,
    vlim=[1e12, 1e20],
    cbartitle=r"$\Sigma$({}) [${}$]".format(line["ion"], pg.Unit("cm**-2").latex()),
    ax=axs[0, 0],
    **pltargs
)

pg.plotting.image(
    s.gas,
    qty="temp",
    av=line["ion"],
    surface_dens=False,
    units="K",
    vlim=[10**3.7, 10**5.2],
    cmap="hot",
    fontcolor="k",
    ax=axs[0, 1],
    **pltargs
)

pg.plotting.image(
    s.gas,
    qty="vel[:,%d]" % zaxis,
    av=line["ion"],
    logscale=False,
    surface_dens=False,
    units="km/s",
    vlim=[-100, 100],
    cbartitle="l.o.s. velocity [km/s]",
    cmap="RdBu",
    fontcolor="k",
    ax=axs[1, 0],
    **pltargs
)

# create a velocity dispersion plot
grid = pg.binning.SPH_to_2Dgrid_by_particle(
    s.gas, qty="vel[:,%d]" % zaxis, av=line["ion"], reduction="stddev", **pltargs
)
vlim = pg.UnitArr([0.0, 80.0], grid.units)
cmap = "plasma"
fig, ax, im = pg.plotting.show_image(
    grid, vlim=vlim, extent=pltargs["extent"], cmap=cmap, ax=axs[1, 1]
)
pltargs.pop("Npx")  # not a keyword in `make_scale_indicators`
pg.plotting.make_scale_indicators(ax, scaleind="line", outline=True, **pltargs)
pg.plotting.add_cbar(
    ax,
    cbartitle=r"$\sigma(v)\,[%s]$" % vlim.units.latex(),
    clim=vlim,
    cmap=cmap,
    fontcolor="w",
    fontoutline=True,
)

fig.tight_layout()

for ax, c in zip(axs.flatten(), ["w", "k", "k", "k"]):
    for i, los in enumerate(loss):
        circle = plt.Circle(
            tuple(los),
            5,
            color="w" if (ax is axs[1, 0] and i in [3, 4]) else c,
            fill=False,
        )
        ax.add_artist(circle)
        ax.annotate(
            str(i + 1),
            xy=los,
            xytext=(7, 7),
            ha="right",
            color=c,
            textcoords="offset points",
        )
fig.savefig("absorption_spectra_los.png", dpi=300, bbox_inches="tight")
# %% [markdown]
# And now let's calculate the actual spectra and their equivalent widths (EW):

# %%
v_limits = pg.UnitArr([-500, 500], "km/s")
fig, ax = plt.subplots(figsize=(15, 4))

for i, los in enumerate(loss):
    # we arenot using all of the calculated properties (see the documentation)
    tau, dens, temp, v_edges, _ = pg.analysis.mock_absorption_spectrum_of(
        s,
        los,
        line_name,
        v_limits,
        xaxis=xaxis,
        yaxis=yaxis,
        method="particles",
    )

    # v_edges are the edges of the bins in velocity space (rest frame velocity
    # that is) convert to observed wavelengths:
    z_edges = pg.analysis.velocities_to_redshifts(v_edges, z0=s.redshift)
    l_edges = pg.UnitScalar(line["l"]) * (1.0 + z_edges)

    EW = pg.analysis.absorption_spectra.EW(tau, l_edges).in_units_of("Angstrom")
    x = (l_edges[:-1] + l_edges[1:]) / 2.0
    ax.plot(
        x,
        np.exp(-tau),
        label=r"l.o.s. #%d   ( EW = %.3f $%s$ )" % (i + 1, float(EW), EW.units.latex()),
        linewidth=3,
    )

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
ax.grid(True)
ax.hlines(1.0, x.min(), x.max())
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([0, 1.1])
ax.set_xlabel(r"wavelength [$%s$]" % x.units.latex(), fontsize=16)
ax.set_ylabel(r"relative flux", fontsize=16)
ax.legend(loc="lower left");
fig.savefig("absorption_spectra.png", dpi=300, bbox_inches="tight")
# %% [markdown]
# ### 2.6.5 ... and much more!

# %% [markdown]
# And there are much more functions for quantitative in `pygad.analysis`. For example:
# 1. NFW profile fitting (`NFW_fit`)
# 2. evaluating SPH fields on given postions (`SPH_qty_at` and `kernel_weighted`)
# 3. calculating X-ray luminosity (`x_ray_luminosity`)
# 4. calculating flow rates (`shell_flow_rates` and `flow_rates`)
# 5. global line-of-sight velocity dispersions `los_velocity_dispersion`
# 6. [...]

# %% [markdown]
# Go and explore!


