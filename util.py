# -*- coding: utf-8 -*-

import numpy as np

def _extend_dim(array):
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
    
    if np.array(wave).ndim > 1:
        wave_array = wave_array_nd(wave, sampling_type, size)
        return wave_array
    
    else:
        wave_array = wave[0] + np.arange(size, dtype = np.double)*wave[1]

        if sampling_type == 'linear':
            return wave_array
        elif sampling_type == 'log':
            wave_array = np.double(10.)**wave_array
            return wave_array
        elif sampling_type == 'ln':
            wave_array = np.e**wave_array
            return wave_array
        
def wave_array_nd(wave, sampling_type, size):
    wave = np.array(wave)
    wave = _extend_dim(wave)
    
    wave_array = np.empty((size,) + wave[0, ...].shape)
    for i, j in np.ndindex(wave[0, ...].shape):
        wave_array[:, i, j] = build_wave_array(wave[:, i, j],
                                               sampling_type,
                                               size)

    wave_array = _reduce_dim(wave_array)
    return wave_array


def fit_wave_interval(wave, sampling_type, size):
    
    assert sampling_type in ['linear', 'log', 'ln']
    
    if sampling_type == 'linear':
        wave_array = np.linspace(wave[0], wave[1], size, dtype = np.double)
    elif sampling_type == 'log':
        wave_array = np.logspace(np.log10(wave[0]), np.log10(wave[1]), size,
                                  np.double(10.), dtype = np.double)
    elif sampling_type == 'ln':
        wave_array = np.logspace(np.log(wave[0]), np.log(wave[1]), size, np.e,
                                  dtype = np.double)
    return wave_array