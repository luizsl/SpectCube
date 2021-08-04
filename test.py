# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from spectrum import Spectrum
from resampling import resampling, _reduce_dim, _extend_dim
from util import build_wave_array
from spectres import spectres
# from astropy.io import fits
from time import perf_counter as clock

#%% Reading data to tests


# #######################################################              test

# obs_file = '../workflow/data/NGC613/Muse/NGC0613_DATACUBE_FINAL_clean.fits'

# #########################################################################


# with fits.open(obs_file) as hdu:
#     data = hdu['DATA'].data
#     error_d = hdu['STAT'].data

#     flux = data[:, 103:106, 103:105]
#     error = error_d[:, 103:106, 103:105]

#     first_wave = hdu['DATA'].header['CRVAL3']
#     step_wave = hdu['DATA'].header['CD3_3']

#     obs = Spectrum(flux, wave = [first_wave, step_wave],
#                    medium = 'air', sampling_type = 'linear', flux_unc = error)


# wave_ln = np.e**(np.arange(np.log(obs.wave[0]),
#                            np.log(obs.wave[-1]),
#                            1*np.log(obs.wave[1]/obs.wave[0])))


#%% Produce artificial signal
size = 100
wave_wcs = [[100, 100],[1,1]]
wave = build_wave_array(wave_wcs, 'linear', size)
flux = np.sin(wave)

plt.plot(wave[:, 0], flux[:, 0])
plt.plot(wave[:, 1], flux[:, 1])

#%%
# flux = data[:, 0:, 103:105].reshape((-1,))
# old_wave = build_wave_array([4750,9300], sampling_type = 'linear', size = len(flux))
# new_wave = build_wave_array([4750,9300], sampling_type = 'ln', size = len(flux))

# t1 = clock()
# resampling(flux = flux,
#             old_wave = old_wave, old_sampling_type = 'linear',
#             new_wave = new_wave, new_sampling_type = 'log')
# print((clock() - t1)*1000)


# t2 = clock()
# spectres(_reduce_dim(flux), _reduce_dim(old_wave), _reduce_dim(new_wave))
# print((clock() - t2)*1000)

#%%

# flux = data[:, :, :]
# old_wave = obs.wave
# new_wave = wave_ln

# t1 = clock()
# resampling(flux = flux,
#             old_wave = old_wave, old_sampling_type = 'linear',
#             new_wave = new_wave, new_sampling_type = 'log')
# print((clock() - t1))
