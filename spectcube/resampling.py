"""
Created on Wed Jul  7 18:52:18 2021

@author: Luiz Albérico
"""

import numpy as np
from scipy import interpolate
# from .util import _extend_dim, _reduce_dim
from spectcube.util import _extend_dim, _reduce_dim


def _build_edges_nd(wave, sampling_type):
    """
    Currently not used

    Parameters
    ----------
    wave : numpy.ndarray
        Array with the wavelengths.
    sampling_type : string
        Sampling type of the array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).

    Returns
    -------
    edges : numpy.ndarray
        Array with the edges of the wavelength bins.

    """
    if sampling_type == 'linear':
        step = wave[[1], ...] - wave[[0], ...]
        edges = wave[[0], ...] - step/2.
        edges = np.append(edges, wave + step/2., axis = 0)
    elif sampling_type == 'log':
        step = np.log10(wave[[1], ...]/wave[[0], ...])
        edges = wave[[0], ...] / 10.**(step/2.)
        edges = np.append(edges, wave * 10.**(step/2.), axis = 0)
    elif sampling_type == 'ln':
        step = np.log(wave[[1], ...]/wave[[0], ...])
        edges = wave[[0], ...] / np.e**(step/2.)
        edges = np.append(edges, wave * np.e**(step/2.), axis = 0)
    return edges

def _build_edges(wave, sampling_type):
    """
    Calculates edges of bins of given wavelength given the center value of the
    bin and the type of sampling.

    Parameters
    ----------
    wave : numpy.ndarray
        Array with the wavelengths.
    sampling_type : string
        Sampling type of the array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).

    Returns
    -------
    edges : numpy.ndarray
        Array with the edges of the wavelength bins.

    """
    if sampling_type == 'linear':
        step = wave[1] - wave[0]
        edges = wave[0] - step/2.
        edges = np.append(edges, wave + step/2.)
    elif sampling_type == 'log':
        step = np.log10(wave[1]/wave[0])
        edges = wave[0] / 10.**(step/2.)
        edges = np.append(edges, wave * 10.**(step/2.))
    elif sampling_type == 'ln':
        step = np.log(wave[1]/wave[0])
        edges = wave[0] / np.e**(step/2.)
        edges = np.append(edges, wave * np.e**(step/2.))
    return edges

def _resampling(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):
    """
    Main function to perform the resampling given the array with the fluxes,
    their sampling wavelengths and sampling type. In addition to the array with
    values of the new wavelength after resampling and the new type of sampling.

    Parameters
    ----------
    flux : numpy.ndarray
        Array with fluxes.
    old_wave : numpy.ndarray
        Array with the wavelength for each flux value.
    old_sampling_type : string
        Sampling type of the input array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    new_wave : numpy.ndarray
        Array with the wavelength for each new flux value.
    new_sampling_type : string
        Sampling type of the output array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    flux_err : numpy.ndarray, optional
        Flux uncertainty (standard deviation). The default is None.

    Returns
    -------
    new_flux : numpy.ndarray
        New array with fluxes after resampling.
    new_flux_err : numpy.ndarray
        Array with uncertainty in the form of standard deviation. If the
        uncertainties are not passed returns None.
    """
    # edges
    old_edges = _build_edges(old_wave, old_sampling_type)
    new_edges = _build_edges(new_wave, new_sampling_type)

    # intervals
    old_inter = np.ediff1d(old_edges)
    new_inter = np.ediff1d(new_edges)

    # integrate and resample the spectrum
    int_flux = np.append([0], np.cumsum(flux * old_inter))

    f_interp = interpolate.interp1d(old_edges, int_flux, bounds_error = False)

    new_flux = f_interp(new_edges)
    new_flux = np.ediff1d(new_flux) / new_inter

    # if the uncertainty is provided it's also processed
    if flux_err is not None:
        int_err = np.append([0], np.cumsum(flux_err * old_inter))

        e_interp = interpolate.interp1d(old_edges, np.square(int_err),
                                        bounds_error = False)

        new_flux_err = np.sqrt(e_interp(new_edges))
        new_flux_err = np.ediff1d(new_flux_err) / new_inter

        return new_flux, new_flux_err
    else:
        return new_flux

def _resampling_nd(flux, old_wave, old_sampling_type, new_wave,
                  new_sampling_type, flux_err = None):
    """
    Private function to perform resampling by calling the _resampling function.

    Parameters
    ----------
    flux : numpy.ndarray
        Array with fluxes.
    old_wave : numpy.ndarray
        Array with the wavelength for each flux value.
    old_sampling_type : string
        Sampling type of the input array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    new_wave : numpy.ndarray
        DESCRIPTION.
    new_sampling_type : string
        Sampling type of the output array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    flux_err : numpy.ndarray, optional
        Flux uncertainty (standard deviation). The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    new_flux : numpy.ndarray
        New array with fluxes after resampling.
    new_flux_err : numpy.ndarray
        Array with uncertainty in the form of standard deviation. If the
        uncertainties are not passed returns None.

    """
    new_flux = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)

    if old_wave.shape[1:] == (1, 1):
        if flux_err is not None:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave, old_sampling_type,
                                new_wave, new_sampling_type,
                                flux_err = flux_err[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                _reduce_dim(old_wave), old_sampling_type,
                                _reduce_dim(new_wave), new_sampling_type)
            return new_flux


    elif old_wave.shape[1:] > (1, 1):
        if flux_err is not None:
            new_flux_err = np.zeros(shape = (new_wave.shape[0],) + flux[0,...].shape)
            for i,j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j], new_flux_err[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave, new_sampling_type,
                                flux_err = flux_err[:, i, j])
            return new_flux, new_flux_err
        else:
            for i, j in np.ndindex(flux[0,...].shape):
                new_flux[:, i, j] = \
                    _resampling(flux[:, i, j],
                                old_wave[:, i, j], old_sampling_type,
                                new_wave, new_sampling_type)
            return new_flux

    else:
        raise TypeError

def resampling(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
               flux_err = None):
    """
    Function for Spectral resampling. Examples of its application can be found
    in https://github.com/luizsl/SpectCube/tree/main/spectcube/example

    Parameters
    ----------
    flux : numpy.ndarray
        Array with fluxes.
    old_wave : numpy.ndarray
        Array with the wavelength for each flux value.
    old_sampling_type : string
        Sampling type of the input array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    new_wave : numpy.ndarray
        DESCRIPTION.
    new_sampling_type : string
        Sampling type of the output array. It can be either linear ('linear'),
        natural logarithm ('ln') or base 10 logarithm (log).
    flux_err : numpy.ndarray, optional
        Flux uncertainty (standard deviation). The default is None.

    Returns
    -------
    new_flux : numpy.ndarray
        New array with fluxes after resampling.
    new_wave : numpy.ndarray
        New array with wavelengths in which streams were sampled.
    new_flux_err : numpy.ndarray
        Array with uncertainty in the form of standard deviation. If the
        uncertainties are not passed returns None.

    """
    assert old_sampling_type and new_sampling_type in ['linear', 'log', 'ln']

    flux = _extend_dim(flux)
    old_wave = _extend_dim(old_wave)
    new_wave = _extend_dim(new_wave)

    if flux_err is None:
        new_flux = _resampling_nd(flux,
                                  old_wave, old_sampling_type,
                                  new_wave, new_sampling_type)
        new_flux = _reduce_dim(new_flux)
        new_wave = _reduce_dim(new_wave)
        return new_flux, new_wave, None
    else:
        flux_err = _extend_dim(flux_err)

        new_flux, new_flux_err = _resampling_nd(flux,
                                                old_wave, old_sampling_type,
                                                new_wave, new_sampling_type,
                                                flux_err = flux_err)
        new_flux = _reduce_dim(new_flux)
        new_wave = _reduce_dim(new_wave)
        new_flux_err = _reduce_dim(new_flux_err)
        return new_flux, new_wave, new_flux_err
