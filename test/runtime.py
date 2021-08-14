# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from resampling import resampling, _reduce_dim, _extend_dim
from util import build_wave_array
from spectres import spectres
from time import perf_counter as clock

def measure_runtime(flux, old_wave, old_sampling_type, new_wave, new_sampling_type,
                    flux_err = None, rep = 10):
    
    runtime = np.empty(rep)
    
    for i in range(rep):
        t1 = clock()
        resampling(flux = flux,
                    old_wave = old_wave, old_sampling_type = old_sampling_type,
                    new_wave = new_wave, new_sampling_type = new_sampling_type)
        runtime[i] = (clock() - t1)*1000
        
    return np.mean(runtime), np.std(runtime)

def build_artificial_signal(wave, sampling_type, size):
    
    if sampling_type == 'linear':
        wave_array = np.linspace(wave[0], wave[1], size, dtype = np.double)
    elif sampling_type == 'log':
        wave_array = np.logspace(np.log10(wave[0]), np.log10(wave[1]), size,
                                 np.double(10.), dtype = np.double)
    elif sampling_type == 'ln':
        wave_array = np.logspace(np.log(wave[0]), np.log(wave[1]), size, np.e,
                                 dtype = np.double)

    flux = np.sin(np.linspace(0, 6*np.pi, size))
    
    return flux, wave_array

##############################################################################
#                       Produce artificial signal
#
# Create artifical signal f(x) with 1000 points (pixels) linearly spaced, 
# with 1 < x < 100.
##############################################################################

size = 1000
wave_lin = np.linspace(1, size, size)
signal = np.sin(np.linspace(0, 6*np.pi, size))

##############################################################################
#                               Runtime 
##############################################################################

# Increasing number of output pixels
array_pixel = np.array([1e2, 1e4, 1e5, 5e5, 2e6, 5e6, 8e6, 1e7],
                       dtype = int)
n_pixel = len(array_pixel)
run_med_pixel = np.zeros(n_pixel)
run_std_pixel = np.zeros(n_pixel)

flux = signal

for index, i in enumerate(array_pixel):
    wave = np.logspace(0, np.log(100), i, base = np.e)
    print(f"Number of pixels: {len(wave)}")
    
    run_med_pixel[index], run_std_pixel[index] = \
    measure_runtime(flux = flux, 
                    old_wave = wave_lin,
                    old_sampling_type = 'linear',
                    new_wave = wave,
                    new_sampling_type = 'ln',
                    rep = 1_000)


# Runtime test for increasing number of spectrums
array_spectra = np.array([1, 1e3, 2e4, 4e4, 6e4, 8e4, 1e5],
                       dtype = int)
n_spectra = len(array_spectra)
run_med_spectra = np.zeros(n_spectra)
run_std_spectra = np.zeros(n_spectra)

wave = np.logspace(0, np.log(100), int(10**3), base = np.e)

for index, j in enumerate(array_spectra):
    print(f"Number of spectrums: {j}")
    if j == 0:
        flux = signal
    if j > 0:
        flux = np.tile(signal, (j, 1)).T

    run_med_spectra[index], run_std_spectra[index] = \
    measure_runtime(flux = flux, 
                    old_wave = wave_lin,
                    old_sampling_type = 'linear',
                    new_wave = wave,
                    new_sampling_type = 'ln',
                    rep = 10)

##############################################################################
#                               Plot
##############################################################################

# Setting aesthetics
plt.style.use('fig_conf.mplstyle')


fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot for increasing number of output pixels
ax1.scatter(array_pixel, run_med_pixel, label = 'Average runtime', 
            color = 'navy')

ax1.fill_between(array_pixel, run_med_pixel-run_std_pixel,
                run_med_pixel+run_std_pixel,
                color = 'Grey',alpha = 0.2,
                label = r'$\sigma$ runtime')

# linear function to comparison
fit_linear = np.polyfit(array_pixel, run_med_pixel, 1)
O_n = np.poly1d(fit_linear)
ax1.plot(array_pixel, O_n(array_pixel) , label = 'O(n)', 
        color = 'darkred', ls = 'dashed')

ax1.set_xlabel('Number of output points')
ax1.set_ylabel('Runtime (ms)')
ax1.legend()

# Plot for increasing number of spectrums
ax2.scatter(array_spectra, run_med_spectra/1e3, label = 'Average runtime',
            color = 'navy')
ax2.fill_between(array_spectra,
                (run_med_spectra-run_std_spectra)/1e3,
                (run_med_spectra+run_std_spectra)/1e3,
                color = 'Grey',alpha = 0.2,
                label = r'$\sigma$ runtime')

# Linear function to comparison
fit_linear = np.polyfit(array_spectra, run_med_spectra/1e3, 1)
O_n = np.poly1d(fit_linear)
ax2.plot(array_spectra, O_n(array_spectra) , label = 'O(n)', 
        color = 'darkred', ls = 'dashed')

ax2.set_xlabel('Number of spectrums')
ax2.set_ylabel('Average runtime (s)')
ax2.legend()

fig.align_labels()

plt.savefig('../doc/figures/runtime.pdf')
