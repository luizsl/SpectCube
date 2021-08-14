#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:24:18 2021

@author: chess-lin
"""

import matplotlib.pyplot as plt
from resampling import resampling
import util as util
from spectres import spectres
from astropy.io import fits


##############################################################################
#                              Reading spectrum
##############################################################################

spec_path = '../data/spec-1665-52976-0519.fits'

with fits.open(spec_path) as hdu:
    obs_unit = hdu['primary'].header['bunit']
    obs_flux = hdu['coadd'].data['flux']
    obs_lam = 10.**hdu['coadd'].data['loglam']

# output wave axis
new_lam = util.fit_wave_interval(wave = [obs_lam[0], obs_lam[-1]], 
                                 sampling_type = 'linear',
                                 size = len(obs_flux))

##############################################################################
#                               Resampling
##############################################################################

# Resampling with SpeCube
our, _ = specube_data = resampling(flux = obs_flux,
                                old_wave = obs_lam,
                                old_sampling_type = 'log',
                                new_wave = new_lam,
                                new_sampling_type = 'linear')

# Resampling with Spectres
spectres_data = spectres(new_lam, obs_lam, obs_flux)

residual_res = ((spectres_data - our)/our)

##############################################################################
#                               Plot
##############################################################################

# Additional aesthetics setting
plt.style.use('fig_conf.mplstyle')
plt.rcParams.update({'lines.linewidth' : 0.8})

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(obs_lam, obs_flux, label = "SDSS", color = 'black')
ax1.plot(new_lam, spectres_data, label = r"$\texttt{SpectRes}$", color = 'navy')
ax1.plot(new_lam, our, label = "Our", color = 'darkred')
ax1.set_ylabel(r'f$_{\lambda}$ ($10^{-17}$ erg/cm$^{2}$/s/$\AA$)')
ax1.set_ylim(10, 210)

axins = ax1.inset_axes([0.5, 0.1, 0.4, 0.4])
x1, x2, y1, y2 = 4330, 4410, 40, 120
axins.plot(obs_lam, obs_flux, color = 'black')
axins.plot(new_lam, spectres_data, color = 'navy')
axins.plot(new_lam, our, color = 'darkred')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yticklabels('')
ax1.axvspan(x1, x2, alpha=0.3, color='Grey')
ax2.axvspan(x1, x2, alpha=0.3, color='Grey')

ax1.legend(loc = 'upper left', framealpha = 1)


ax2.plot(new_lam, 1e4*residual_res, label = 'Residuals', color = 'black')
ax2.set_ylabel(r'Diff. ($\times 10^{-4}$)')
ax2.set_xlabel(r'$\AA$')
ax2.set_ylim(-2.5, 2.5)

fig.align_ylabels()

plt.savefig('../doc/figures/comparison.pdf')
