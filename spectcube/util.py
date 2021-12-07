"""
@author: Luiz Albérico
"""
# __all__ = []

import numpy as np

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
        array = array[..., np.newaxis, np.newaxis]
    elif array.ndim == 2:
        array = array[..., np.newaxis]
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
    wave = np.array(wave)
    wave = _extend_dim(wave)

    wave_array = np.empty((size,) + wave[0, ...].shape)
    for i, j in np.ndindex(wave[0, ...].shape):
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

    if np.array(wave).ndim > 1:
        wave_array = _wave_array_nd(wave, sampling_type, size)
        return wave_array

    wave_array = wave[0] + np.arange(size)*wave[1]

    if sampling_type == 'linear':
        return wave_array
    if sampling_type == 'log':
        wave_array = 10.**wave_array
        return wave_array
    if sampling_type == 'ln':
        wave_array = np.e**wave_array
        return wave_array

    raise TypeError

def fit_wave_interval(wave, old_sampling, new_sampling, new_size = None):
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
    # global old_edge, step, lower_edge, upper_edge
    assert old_sampling and new_sampling in ['linear', 'log', 'ln']
    
    if new_size is None:
        new_size = len(wave)
    # wave = wave_model
    # old_sampling = 'linear'
    # new_size = 4300
    old_edge = _build_edges(wave = wave, sampling_type = old_sampling)
    lower_edge = old_edge[0]
    upper_edge = old_edge[-1]
        
    if new_sampling == 'linear':
        step = (upper_edge - lower_edge) / (new_size)
        
        wave_array = np.linspace(lower_edge + step/1.99,
                                 upper_edge - step/1.99,
                                 num = new_size)
    elif new_sampling == 'log':
        lower_edge = np.log10(lower_edge)
        upper_edge = np.log10(upper_edge)
        step = (upper_edge - lower_edge) / (new_size)
        
        wave_array = np.logspace(lower_edge + step/1.99,
                                 upper_edge - step/1.99,
                                 num = new_size, base = 10.)
    elif new_sampling == 'ln':
        lower_edge = np.log(lower_edge)
        upper_edge = np.log(upper_edge)
        step = (upper_edge - lower_edge) / (new_size)
        
        wave_array = np.logspace(lower_edge + step/1.99,
                                 upper_edge - step/1.99,
                                 num = new_size, base = np.e)

    # wave_array[[0,-1]] = wave[0], wave[1]
    return wave_array

# novo = fit_wave_interval(wave= wave_model, old_sampling = 'linear', new_sampling = 'log')
