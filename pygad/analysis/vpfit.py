"""
Fit a spectrum with Voigt profiles, adding lines until the desired chisq value is achieved.  

Doctests:
    >>> import pygad as pg
    >>> ion = 'H1215'
    >>> s = pg.Snapshot(snap_file)
    >>> L = s.boxsize.in_units_of('Mpc', subs=s)
    >>> H = s.cosmology.H(s.redshift).in_units_of('(km/s)/Mpc', subs=s)
    >>> redshift = s.redshift
    >>> box_width = s.boxsize.in_units_of('Mpc', subs=s)
    >>> los = L.in_units_of('kpc') * np.random.rand(1,2)  # random LOS
    >>> lambda_rest = float(pg.analysis.absorption_spectra.lines[ion]['l'].split()[0])
    >>> spec_name = 'spec'+snap_file
    >>> vbox = H * box_width
    >>> v_limits = [0, vbox]
    >>> taus, col_densities, phys_densities, temps, metallicities, vpec, v_edges, restr_column = pg.analysis.absorption_spectra.mock_absorption_spectrum_of(s, los, line_name, v_limits, Nbins=(1+int(periodic_vel))*Nbins, return_los_phys=True)
    >>> velocities = 0.5 * (v_edges[1:] + v_edges[:-1])
    >>> wavelengths = lambda_rest * (s.redshift + 1) * (1 + velocities / c)
    >>> sigma_noise = 0.1  # add noise with this sigma (this gives S/N=10 per pixel)
    >>> noise = np.random.normal(0.0, sigma_noise, len(wavelengths))
    >>> noise_vector = np.asarray([sigma_noise] * len(noise))
    >>> fluxes = np.exp(-np.array(taus)) + noise
    >>> flux,noise_vector = pg.analysis.apply_LSF(wavelengths, fluxes, noise_vector, grating='COS_G130M')  # smooth with desired line spread fcn
    >>> contin = pg.analysis.fit_continuum(wavelengths, fluxes, noise_vector, order=0)  # do continuum fitting
    >>> fluxes = fluxes / contin
    >>> noise_vector  = noise_vector / contin
    >>> pg.analysis.write_spectrum(spec_name, los, lambda_rest, s.redshift, velocities, fluxes, taus, noise_vector, col_densities, phys_densities, temps, metallicities, vpec) # write spectrum
    >>> line_list = pg.analysis.fit_profiles(ion, lam, flux, noise, chisq_lim=2.0, max_lines=7, logN_bounds=[12,19], b_bounds=[3,100], mode='Voigt')
    >>> pg.analysis.write_lines(spec_name, line_list, 0) # append line info to spectrum file
    >>> model_flux, N, dN, b, db, l, dl, EW = pg.analysis.plot_fit(ax[ilos], lam, flux, noise, line_list, ion, starting_pixel=istart, show_plot=True)  # plot spectrum and lines
"""

__all__ = [
    "fit_profiles",
    "model_tau",
    "find_regions",
    "EquivalentWidth",
    "periodic_wrap",
    "periodic_unwrap_wavelength",
    "write_spectrum",
    "write_lines",
    "plot_fit",
]

from ..units import UnitArr
from ..physics import c

# from .. import utils
from .. import environment
import numpy as np
import pylab as plt

from .absorption_spectra import lines, Gaussian, Lorentzian, Voigt, line_profile


def fit_profiles(
    line,
    l,
    flux,
    noise,
    chisq_lim=2.0,
    max_lines=10,
    mode="Voigt",
    logN_bounds=[12, 19],
    b_bounds=[5, 200],
):
    """
    Fit Voigt/other profiles to the given spectrum.  Begins with one
    line, then adds lines until desired chi-sq is achieved.

    Args:
        line (str):         The line to fit as listed in
                            `analysis.absorption_spectra.lines`, e.g. 'HI1215'.
        l (array-like):     The wavelengths of the input spectrum to fit.
        flux (array-like):  The normalized flux at the given wavelengths,
                            i.e. the spectrum to fit.
        noise (array-like): The normalized 1-sigma noise vector at the given
                            wavelengths.  Must always be >0.
        chisq_lim (float):  Number of sigma below which chi-sq is considered
                            to be a "good fit" and no more lines are added.
                            If <0, then the value used is abs(chisq_lim)+0.1*n_lines,
                            where n_lines is the number of lines for that trial.
        max_lines (int):    Maximum number of lines allowed in a given detection
                            region, after which the fit declared done regardless of chisq.
                            If limit is hit, this may result in a poor fit.
        mode (str):         Type of line profile: Gaussian/Lorentzian/Voigt;
                            see absorption_spectra.line_profile().
        logN_bounds (list): Initial log(column density) is restricted to this range
        b_bounds (list):    Initial line width is restricted to this range (km/s)


    Returns:
        profiles:    Dictionary of [N, dN, b, db, l, dl, EW] of best-fit
                     profiles.
        tau_model:   Optical depths of best-fit model.

    """

    if isinstance(line, str):
        line = lines[line]
    l0 = line["l"]
    if isinstance(l, np.ndarray) or l.units in [
        1,
        None,
    ]:  # set units of l to Angstrom if none supplied
        l = UnitArr(l, "Angstrom")

    def _tau_to_flux(tau):  # return flux from tau, avoiding over/underflow
        return np.exp(-np.clip(tau, -50, 50))

    def _chisq(p, l, flux, noise, mode):  # reduced chisq
        return np.sum(
            (_tau_to_flux(model_tau(line, p, l, mode)) - flux) ** 2 / noise ** 2
        ) / len(l)

    def _add_line(p, bnd, l, flux, l0, mode):  # adds N, b, l for a new line
        l_bounds = [l[1], l[-2]]
        b_guess = (
            (l_bounds[1] - l_bounds[0]) / float(l0.split()[0]) * 3.0e5 / 5.0
        )  # first guess at b
        b_guess = max(2 * b_bounds[0], 0.5 * min(b_bounds[1], b_guess))
        if len(p) == 0:
            resid = flux
            n_guess = 14.0 - resid[np.argmin(resid)]  # first guess at logN
            p = np.array([n_guess])  # rough guess of logN
        else:
            resid = (
                1.0 + flux - _tau_to_flux(model_tau(line, p, l, mode))
            )  # residual spectrum
            n_guess = 14.0 - resid[np.argmin(resid)]  # first guess at logN
            p = np.append(p, n_guess)  # rough guess of logN
        p = np.append(p, b_guess)  # first guess of b
        p = np.append(p, l[np.argmin(resid)])  # add line @min of residual flux
        if len(bnd) == 0:
            bnd = np.array([logN_bounds])
        else:
            bnd = np.append(bnd, np.array([logN_bounds]), axis=0)
        bnd = np.append(bnd, np.array([b_bounds]), axis=0)
        bnd = np.append(bnd, np.array([l_bounds]), axis=0)
        # if environment.verbose <= environment.VERBOSE_TALKY:
        #    print('adding line logN=%g (%g) b=%g l=%g (%g-%g)'%(p[-3],min(resid),p[-2],p[-1],l[0],l[-1]))
        return p, bnd

    # identify independent regions to fit within the spectrum
    regions_l, regions_i = find_regions(l, flux, noise, min_region_width=2)

    # dicts to store results
    line_list = {
        "region": np.array([]),
        "l": np.array([]),
        "dl": np.array([]),
        "b": np.array([]),
        "db": np.array([]),
        "N": np.array([]),
        "dN": np.array([]),
        "EW": np.array([]),
    }

    # loop over regions
    from scipy.optimize import minimize

    for ireg in range(len(regions_l)):
        params = []
        bounds = []
        n_lines = 0
        best_nlines = 1
        chisq_old = 1.0e20
        chisq_accept = abs(chisq_lim)
        l_reg = l[regions_i[ireg, 0] : regions_i[ireg, 1]]
        f_reg = flux[regions_i[ireg, 0] : regions_i[ireg, 1]]
        n_reg = noise[regions_i[ireg, 0] : regions_i[ireg, 1]]
        # loop to add lines until desired chisq achieved
        while n_lines < max_lines:
            params, bounds = _add_line(params, bounds, l_reg, f_reg, l0, mode)
            n_lines = int(len(params) / 3)
            chisq_fcn = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, mode),
                options={"maxiter": 100},
            )
            params = soln.x  # set params to new chisq-minimized values
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if chisq_soln < chisq_old:
                chisq_old = chisq_soln
                best_nlines = n_lines
            if environment.verbose >= environment.VERBOSE_TACITURN:
                print(
                    "Region %d: %d lines gives chisq=%g (%g) after %d iters"
                    % (ireg, n_lines, chisq_soln, chisq_accept, soln.nit)
                )
            """
            if params[n_lines*3-1] < l[2] or params[n_lines*3-1] > l[-2]:
                if environment.verbose <= environment.VERBOSE_TACITURN:
                    print('removing edge line %g<%g or %g>%g'%(params[n_lines*3-1], l[2], params[n_lines*3-1], l[-2]))
                params = params[:-3]  # if line is right at the edge, remove it and try again
                params += 0.02*(2*np.random.rand(len(params))-1)  # jiggle params to avoid hitting same solution
                chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            """
            if chisq_soln < chisq_accept:
                break
            if chisq_lim < 0:
                chisq_accept += 0.1

        if chisq_soln > chisq_accept:  # try to go back to previous best solution
            if environment.verbose == environment.VERBOSE_TACITURN:
                print(
                    "Region %d uncoverged with %d lines, chisq=%g; retrying with %d"
                    % (ireg, n_lines, chisq_soln, best_nlines)
                )
            n_lines = best_nlines
            params = params[: 3 * n_lines]
            bounds = bounds[: 3 * n_lines]
            chisq_fcn = lambda *args: _chisq(*args)
            soln = minimize(
                chisq_fcn,
                params,
                bounds=bounds,
                args=(l_reg, f_reg, n_reg, mode),
                options={"maxiter": 100},
            )
            params = soln.x  # set params to new chisq-minimized values
            chisq_soln = _chisq(params, l_reg, f_reg, n_reg, mode)
            if chisq_soln > chisq_accept:
                if environment.verbose >= environment.VERBOSE_TACITURN:
                    print(
                        "WARNING: region %d has large chisq=%g; check fit"
                        % (ireg, chisq_soln)
                    )

        params += 0.02 * (
            2 * np.random.rand(len(params)) - 1
        )  # jiggle params and refit to compute hessian
        soln = minimize(
            chisq_fcn,
            params,
            args=(l_reg, f_reg, n_reg, mode),
            method="BFGS",
            options={"maxiter": 100},
        )
        cov = soln.hess_inv  # covariance matrix of final soluiton

        # append lines in this region onto line list
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                "region %d (%g-%g): chisq= %g with %d lines"
                % (
                    ireg,
                    regions_l[ireg, 0],
                    regions_l[ireg, 1],
                    chisq_soln,
                    int(len(params) / 3),
                )
            )
        for ip in range(n_lines):
            line_list["region"] = np.append(line_list["region"], ireg)
            line_list["N"] = np.append(line_list["N"], params[ip * 3])
            line_list["b"] = np.append(line_list["b"], params[ip * 3 + 1])
            line_list["l"] = np.append(line_list["l"], params[ip * 3 + 2])
            line_list["dN"] = np.append(line_list["dN"], np.sqrt(cov[ip * 3, ip * 3]))
            line_list["db"] = np.append(
                line_list["db"], np.sqrt(cov[ip * 3 + 1, ip * 3 + 1])
            )
            line_list["dl"] = np.append(
                line_list["dl"], np.sqrt(cov[ip * 3 + 2, ip * 3 + 2])
            )
            tau_line = model_tau(
                line,
                [params[ip * 3], params[ip * 3 + 1], params[ip * 3 + 2]],
                l_reg,
                mode,
            )
            line_list["EW"] = np.append(
                line_list["EW"], EquivalentWidth(_tau_to_flux(tau_line), l_reg)
            )

    return line_list


def model_tau(line, p, l, mode="Voigt"):
    """
    Compute optical depth vs. wavelength for a set of lines.

    Args:
        p (numpy array): [logN,b,wavelength] for a set of lines, in
                         a flattened numpy array.
        l (numpy array): Wavelengths over which to compute model spectrum.
    Returns:
        total_tau:  Optical depths (vs. l) from the combined set of lines
    """
    p = np.array(p)
    total_tau = np.zeros(len(l), dtype=float)
    if len(p) == 0:
        return total_tau  # no lines yet, return zeros
    for ip in range(int(len(p) / 3)):
        _, tau = line_profile(
            line, 10 ** p[ip * 3], b=p[ip * 3 + 1], l0=p[ip * 3 + 2], l=l, mode=mode
        )
        total_tau += tau  # add up optical depths from all lines
    return total_tau


def EquivalentWidth(fluxes, waves):
    """
    Find the equivalent width of a line/region.

    Args:
        taus (numpy array): the optical depths.
        waves (numpy array): list of wavelength for region.
    Returns:
        Equivalent width in units of waves.
    """
    dEW = np.zeros(len(fluxes))
    for i in range(1, len(fluxes) - 1):
        dEW[i] = (1.0 - fluxes[i]) * abs(waves[i + 1] - waves[i - 1]) * 0.5
    dEW[0] = (1.0 - fluxes[0]) * abs(waves[1] - waves[0])
    dEW[i - 1] = (1.0 - fluxes[i - 1]) * abs(waves[i - 1] - waves[i - 2])
    return np.sum(dEW)


def find_regions(
    wavelengths, fluxes, noise, min_region_width=2, N_sigma=5.0, extend=False
):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.0] * num_pixels
    noise_ews = [0.0] * num_pixels
    det_ratio = [-float("inf")] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.0
    noise_ews[0] = 0.0

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels - 1) / 2.0 for p in range(num_pixels)])

    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):

        gaussian = np.exp(-0.5 * (xarr / std) ** 2)

        flux_func = np.convolve(flux_ews, gaussian, "same")
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), "same")

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and det_ratio[i] > N_sigma and fluxes[i] < 1.0:
            start = i
        elif start != 0 and (det_ratio[i] < N_sigma or fluxes[i] > 1.0):
            if (i - start) > min_region_width:
                end = i
                region_endpoints.append([start, end])
            start = 0

    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts

    if extend:
        # Expand edges of region until flux goes above 1
        regions_expanded = []
        for reg in region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes) - 1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])

    else:
        regions_expanded = region_endpoints

    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    buffer = 3
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        # TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
        if i < (len(regions_expanded) - 1) and end > regions_expanded[i + 1][0]:
            end = regions_expanded[i + 1][1]
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if flux_dec > abs(noise[j]) * N_sigma:
                if start >= buffer:
                    start -= buffer
                if end < len(wavelengths) - buffer:
                    end += buffer
                regions_l.append([wavelengths[start], wavelengths[end]])
                regions_i.append([start, end])
                break

    if environment.verbose >= environment.VERBOSE_TACITURN:
        print("Found {} detection regions".format(len(regions_l)))
    return np.array(regions_l), np.array(regions_i)


def periodic_wrap(l, flux, noise):
    """
    To avoid situations where the end of a spectrum is in the middle of
    an absorption feature, this routine periodically wraps the spectrum
    such that the endpoint has the highest flux value.  The wavelengths
    are not changed, only the flux and noise vectors are wrapped.
    Should only be used with periodic simulations. Assumes the spectrum
    spans the entire simulation volume.

    Args:
        l (numpy array):     list of wavelength for region.
        flux (numpy array):  fluxes of wavelength for region.
        noise (numpy array): noise array (1sigma)

    Returns:
        flux:  periodically wrapped fluxes
        noise: periodically wrapped noise array
        starting_pixel: the pixel number where the highest flux occurs
    """

    starting_pixel = np.argmax(flux)
    flux = np.concatenate((flux[starting_pixel:-1], flux[0 : starting_pixel + 1]))
    noise = np.concatenate((noise[starting_pixel:-1], noise[0 : starting_pixel + 1]))
    if environment.verbose >= environment.VERBOSE_TACITURN:
        print("Periodically wrapping spectrum, starting_pixel= %d" % starting_pixel)

    return flux, noise, starting_pixel


def periodic_unwrap_wavelength(waves, l, starting_pixel):
    """
    After periodic_wrap(), the lines will have the wrong
    wavelength.  This routine 'unwraps' the wavelengths to
    place the line back where it should be.

    Args:
        waves (numpy array):  Wavelength values to be unwrapped.
        l (numpy array):      Wavelengths of the periodic spectrum.
        starting_pixel (int): The pixel number where the wrapping began.

    Returns:
        wave_line:  Values of wavelengths after unwrapping.
    """

    waves = waves - l[0] + l[starting_pixel]
    waves = np.where(waves > l[-1], waves - (l[-1] - l[0]), waves)  # wrap wavelengths

    return waves


def write_spectrum(
    spec_name,
    line,
    LOS_pos,
    lambda_rest,
    redshift,
    vels,
    fluxes,
    taus,
    noise,
    col_dens,
    phys_dens,
    temps,
    mets,
    vpec,
    overwrite=True,
):
    """
    Output spectrum to hdf5 file.

    Args:
        spec_name (str):      Name of file to write spectrum out to.
                              '.h5' will be appended to this.
        line (str):           The ion name, e.g. 'H1215'
        LOS_pos (list/array): (x,y,z) position of LOS,
                              with the LOS axis holding a value of -1.
        lambda_rest (float):  Rest wavelength of ion
        redshift (float):     Redshift of snapshot
        vels (list/array):    Velocities of pixels.
        fluxes (list/array):  Normalized fluxes of pixels; this should include
                              noise, smoothing, etc. so it's NOT =exp(-taus).
        taus (list/array):    Optical depths of pixels
        noise (list/array):   1-sigma noise array of pixels
        col_dens (list/array): Column densities for each pixel
        phys_dens (list/array): Tau-weighted physical densities for each pixel
        temps (list/array):   Tau-weighted gas temperatures for each pixel
        mets (list/array):    Tau-weighted metal mass fractions for each pixel
        vpec (list/array):    LOS peculiar velocity for each pixel

    Returns:

    """
    import h5py
    import os

    if os.path.isfile(spec_name) and not overwrite:
        if environment.verbose >= environment.VERBOSE_TACITURN:
            print(
                "WARNING: write_spectrum() failed: File %s exists, and overwrite set to False"
                % spec_name
            )
        return

    waves = lambda_rest * (redshift + 1.0) * (1.0 + vels / c)
    mets = np.log10(np.where(mets < 1.0e-10, 1.0e-10, mets))  # turn into log10(Z)
    if len(LOS_pos) == 2:
        LOS_pos = np.append(
            np.array(LOS_pos), -1.0
        )  # assumes if only 2 values are provided, they are (x,y), so we add -1 for z.

    with h5py.File("%s.h5" % spec_name, "w") as hf:
        lam0 = hf.create_dataset("lambda_rest", data=lambda_rest)
        lam0.attrs["ion_name"] = line  # store line name as attribute of rest wavelength
        hf.create_dataset("LOS_pos", data=np.array(LOS_pos))
        hf.create_dataset("redshift", data=redshift)
        hf.create_dataset("velocity", data=np.array(vels))
        hf.create_dataset("wavelength", data=np.array(waves))
        hf.create_dataset("flux", data=np.array(fluxes))
        hf.create_dataset("tau", data=np.array(taus))
        hf.create_dataset("noise", data=np.array(noise))
        hf.create_dataset("col_density", data=np.array(col_dens))
        hf.create_dataset("phys_density", data=np.array(phys_dens))
        hf.create_dataset("temperature", data=np.array(temps))
        hf.create_dataset("metallicity", data=np.array(mets))
        hf.create_dataset("vpec", data=np.array(vpec))

    return


def write_lines(spec_name, line_list, starting_pixel=0):
    """
    Append profile fit information to spectrum file.  Spectrum file must exist
    and contain the ion name attribute and wavelength list.

    Args:
        spec_name (str):      Name of file to write lines out to.
                              '.h5' will be appended to this.
                              Info placed in hdf5 group called 'lines'.
        line_list (dict):     List of fitted absorption feature
                              as output by fit_spectrum().
        starting_pixel (int): The pixel number where the wrapping began.
                              Set to 0 (default) for no periodic wrapping.

    Returns:

    """
    import h5py

    with h5py.File("%s.h5" % spec_name, "r") as hf:
        line = hf["lambda_rest"].attrs["ion_name"]
        waves = np.array(hf["wavelength"])

    # create overall model spectrum from all lines combined
    tau_model = np.zeros(len(waves))
    for i in range(len(line_list["N"])):
        p = np.array([line_list["N"][i], line_list["b"][i], line_list["l"][i]])
        tau_model += model_tau(line, p, waves)
    model_flux = np.exp(-np.clip(tau_model, -30, 30))

    # load data into arrays
    region = line_list["region"]
    N = line_list["N"]
    dN = line_list["dN"]
    b = line_list["b"]
    db = line_list["db"]
    if starting_pixel > 0:
        l = periodic_unwrap_wavelength(line_list["l"], waves, starting_pixel)
    else:
        l = dl = line_list["l"]
    dl = line_list["dl"]
    EW = line_list["EW"]

    with h5py.File("%s.h5" % spec_name, "a") as hf:
        lines = hf.create_group("lines")
        lines.create_dataset("fit_logN", data=np.array(N))
        lines.create_dataset("fit_dlogN", data=np.array(dN))
        lines.create_dataset("fit_b", data=np.array(b))
        lines.create_dataset("fit_db", data=np.array(db))
        lines.create_dataset("fit_l", data=np.array(l))
        lines.create_dataset("fit_dl", data=np.array(dl))
        lines.create_dataset("fit_EW", data=np.array(EW))
        lines.create_dataset("model_flux", data=np.array(model_flux))
        lines.create_dataset("starting_pixel", data=starting_pixel)

    return


def plot_fit(
    ax,
    waves,
    flux,
    noise,
    line_list,
    line,
    starting_pixel=0,
    show_plot=True,
    show_label=True,
):
    """
    Generates mdoel flux and plots spectrum, including the model flux and identified lines.  Returns model flux and line list.
    Args:
        waves (numpy array):  Wavelengths of original spectrum.
        flux (numpy array):   Fluxes of original spectrum.
        noise (numpy array):  Noise vector of original spectrum.
        line_list (dict):     List of fitted absorption feature
                              as output by fit_spectrum().
        line (str):           The ion name, e.g. 'H1215'
        starting_pixel (int): The pixel number where the wrapping began.
                              Set to 0 (default) for no periodic wrapping.

    Returns:
        flux_model:   Flux values of model (fitted) spectrum
        N, dN, b, db, l, dl, EW:  Fit parameters copied from line_list,
                      returned as numpy arrays.
    """
    tau_model = np.zeros(len(waves))
    for i in range(len(line_list["N"])):
        p = np.array([line_list["N"][i], line_list["b"][i], line_list["l"][i]])
        tau_model += model_tau(line, p, waves)
    flux_model = np.exp(-np.clip(tau_model, -30, 30))

    region = line_list["region"]
    N = line_list["N"]
    dN = line_list["dN"]
    b = line_list["b"]
    db = line_list["db"]
    if starting_pixel > 0:
        l = periodic_unwrap_wavelength(line_list["l"], waves, starting_pixel)
    else:
        l = dl = line_list["l"]
    dl = line_list["dl"]
    EW = line_list["EW"]

    if starting_pixel > 0:
        lamorig = periodic_unwrap_wavelength(waves, waves, starting_pixel)
    else:
        lamorig = waves
    sortid = np.argsort(lamorig)

    if show_label:
        ax.plot(lamorig[sortid], flux[sortid], alpha=0.4, label="flux")
        ax.plot(lamorig[sortid], noise[sortid], alpha=0.5, c="y")
        ax.plot(lamorig[sortid], flux_model[sortid], "--", label="model")
        ax.legend(loc="lower right")
    else:
        ax.plot(lamorig[sortid], flux[sortid], alpha=0.4)
        ax.plot(lamorig[sortid], noise[sortid], alpha=0.5, c="y")
        ax.plot(lamorig[sortid], flux_model[sortid], "--")
    for lw in l:
        ax.axvline(x=lw, ymin=0.95, ymax=1)

    if show_plot:
        plt.legend(loc="best")
        plt.show()

    return flux_model, N, dN, b, db, l, dl, EW
