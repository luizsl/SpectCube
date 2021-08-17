# -*- coding: utf-8 -*-

import numpy as _np

def _extend_dim(array):
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

def build_wave_array(wave, sampling_type, size):
    
    assert sampling_type in ['linear', 'log', 'ln']
    
    if _np.array(wave).ndim > 1:
        wave_array = _wave_array_nd(wave, sampling_type, size)
        return wave_array
    
    else:
        wave_array = wave[0] + _np.arange(size, dtype = _np.double)*wave[1]

        if sampling_type == 'linear':
            return wave_array
        elif sampling_type == 'log':
            wave_array = _np.double(10.)**wave_array
            return wave_array
        elif sampling_type == 'ln':
            wave_array = _np.e**wave_array
            return wave_array
        
def _wave_array_nd(wave, sampling_type, size):
    wave = _np.array(wave)
    wave = _extend_dim(wave)
    
    wave_array = _np.empty((size,) + wave[0, ...].shape)
    for i, j in _np.ndindex(wave[0, ...].shape):
        wave_array[:, i, j] = build_wave_array(wave[:, i, j],
                                               sampling_type,
                                               size)

    wave_array = _reduce_dim(wave_array)
    return wave_array


def fit_wave_interval(wave, sampling_type, size):
    
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