"""
@author: Luiz Albérico
"""

import numpy as _np


def _extend_dim(array):
    """
    SpectCube resampling code always works on 3d arrays, if an array with less
    dimensions is used this function adds a new empty axis.

    Parameters
    ----------
    array : numpy.ndarray
        Array with numerical values of 1, 2 or 3 dimensions.

    Raises
    ------
    TypeError
        If the input array has more than 3 dimensions an exception is raised.

    Returns
    -------
    array : numpy.ndarray
        Returns an array with 3 dimensions.

    """
    if array.ndim == 3:
        pass
    elif array.ndim == 1:
        array = array[..., _np.newaxis, _np.newaxis]
    elif array.ndim == 2:
        array = array[..., _np.newaxis]
    else:
        raise TypeError

    return array

def _reduce_dim(array):
    """
    Removes the empty axes added with the _exten_dim function, returning to the
    user an array with the same number of dimensions as the input array.

    Parameters
    ----------
    array : numpy.ndarray
        Array with numerical values of 1, 2 or 3 dimensions.

    Returns
    -------
    array : numpy.ndarray
        Array with numerical values of 1, 2 or 3 dimensions.

    """
    if array.ndim == 3:
        if array.shape[1:] == (1, 1):
            array = array[:, 0, 0]
        elif array.shape[2:] == (1, ):
            array = array[:, :, 0]
    elif array.ndim == 2:
        if array.shape[1:] == (1,):
            array = array[:, 0]
    else:
        pass

    return array

def _wave_array_nd(wave, sampling_type, size):
    """
    Private method to create wavelength array in case of 2D or 3D array.

    Parameters
    ----------
    wave : np.ndarray, list
        List or ndarray with initial wavelength and step e.g.:
        wave = [first_wave, step].
    sampling_type : string
        Spectrum sampling type, use 'linear' if equally spaced linearly,
        'ln' if equally spaced in power of e (Euler number) or 'log' if
        equally spaced in powers of base 10.
    size : integer
        Number of pixels in the wavelength array.

    Returns
    -------
    wave_array : np.ndarray
        Array with wavelength values
    """
    wave = _np.array(wave)
    wave = _extend_dim(wave)

    wave_array = _np.empty((size,) + wave[0, ...].shape)
    for i, j in _np.ndindex(wave[0, ...].shape):
        wave_array[:, i, j] = build_wave_array(wave[:, i, j],
                                               sampling_type,
                                               size)

    wave_array = _reduce_dim(wave_array)
    return wave_array

def build_wave_array(wave, sampling_type, size):
    """
    Creates wavelength array to facilitate application of resampling.

    Parameters
    ----------
    wave : np.ndarray, list
        List or ndarray with initial wavelength and step e.g.:
        wave = [first_wave, step].
    sampling_type : string
        Spectrum sampling type, use 'linear' if equally spaced linearly,
        'ln' if equally spaced in power of e (Euler number) or 'log' if
        equally spaced in powers of base 10.
    size : integer
        Number of pixels in the wavelength array.

    Returns
    -------
    wave_array : np.ndarray
        Array with wavelength values

    Examples
    --------
    To produce a single wavelength array starting at 100 containing 10 elements
    and evenly spaced at 1 (arbitrary units):

    >>> sc.util.build_wave_array([100,1], 'linear', 10)
    array([100., 101., 102., 103., 104., 105., 106., 107., 108., 109.])

    To produce two arrays, one spaced 1 and one spaced 2 (arbitrary units):
    >>> sc.util.build_wave_array([[100,100],[1,2]], 'linear', 10)
    array([[100., 100.],
       [101., 102.],
       [102., 104.],
       [103., 106.],
       [104., 108.],
       [105., 110.],
       [106., 112.],
       [107., 114.],
       [108., 116.],
       [109., 118.]])

    Creating evenly spaced wavelength arrays in natural logarithmic scale:
    >>> wave_array = sc.util.build_wave_array([3,1e-4], 'ln', 10)
    >>> wave_array
    array([20.08553692, 20.08754558, 20.08955443, 20.09156349, 20.09357275,
       20.0955822 , 20.09759186, 20.09960172, 20.10161178, 20.10362204])

    Note that the log of this wavelength array is evenly spaced:
    >>> np.log(wave_array)
    array([3.    , 3.0001, 3.0002, 3.0003, 3.0004, 3.0005, 3.0006, 3.0007,
           3.0008, 3.0009])
    """
    assert sampling_type in ['linear', 'log', 'ln']

    if _np.array(wave).ndim > 1:
        wave_array = _wave_array_nd(wave, sampling_type, size)
        return wave_array

    wave_array = wave[0] + _np.arange(size, dtype = _np.double)*wave[1]

    if sampling_type == 'linear':
        return wave_array
    if sampling_type == 'log':
        wave_array = _np.double(10.)**wave_array
        return wave_array
    if sampling_type == 'ln':
        wave_array = _np.e**wave_array
        return wave_array

    raise TypeError

def fit_wave_interval(wave, sampling_type, size):
    """
    Produces an array of wavelengths between two values and with a given number
    of elements.

    Parameters
    ----------
    wave : np.ndarray, list
        List or ndarray with initial wavelength and final wavelength e.g.:
        wave = [first_wave, last_wave].
    sampling_type : string
        Spectrum sampling type, use 'linear' if equally spaced linearly,
        'ln' if equally spaced in power of e (Euler number) or 'log' if
        equally spaced in powers of base 10.
    size : integer
        Number of pixels in the wavelength array.

    Returns
    -------
    wave_array : np.ndarray
        Array with wavelength values

    Examples
    --------
    To produce an array of wavelengths between 3000 and 3100 (arbitrary units)
    with 10 elements and equally spaced.
    >>> sc.util.fit_wave_interval([3000,3100], 'linear', 10)
    array([3000.        , 3011.11111111, 3022.22222222, 3033.33333333,
           3044.44444444, 3055.55555556, 3066.66666667, 3077.77777778,
           3088.88888889, 3100.        ])

    To produce the same array but equally spaced in base 10 logarithms.
    >>> sc.util.fit_wave_interval([3000,3100], 'log', 10)
    array([3000.        , 3010.94987574, 3021.93971808, 3032.96967289,
           3044.03988657, 3055.15050608, 3066.30167889, 3077.49355302,
           3088.72627702, 3100.        ])
    """
    assert sampling_type in ['linear', 'log', 'ln']

    if sampling_type == 'linear':
        wave_array = _np.linspace(wave[0], wave[1], size, dtype = _np.double)
    elif sampling_type == 'log':
        wave_array = _np.logspace(_np.log10(wave[0]), _np.log10(wave[1]),
                                  num = size, base = _np.double(10.),
                                  dtype = _np.double)
    elif sampling_type == 'ln':
        wave_array = _np.logspace(_np.log(wave[0]), _np.log(wave[1]),
                                  num = size, base = _np.e, dtype = _np.double)
    return wave_array
