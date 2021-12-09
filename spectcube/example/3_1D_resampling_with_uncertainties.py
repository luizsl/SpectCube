import spectcube as sc
import matplotlib.pyplot as plt

from astropy.io import fits
from os import path

def run_example_3():
    # Resampling SDDS spectrum from linear to natural logarithm. One should always
    # avoid to resample the observations preferring resampling the models which are
    # noise free. However in some cases this may be unavoidable. Here we resample
    # the observations and also process the uncertainties.
    
    # setting path to the module directory
    pack_dir = path.dirname(path.realpath(sc.__file__))
    
    # Path to observation data
    obs_path = pack_dir + '/data/spec-1665-52976-0519.fits'
    
    # Read observation data
    with fits.open(obs_path) as hdu:
        obs_flux = hdu['COADD'].data['flux']
        obs_err = (1.0/hdu['COADD'].data['ivar'])**0.5
        obs_loglam = 10**hdu['COADD'].data['loglam']
    
    # At this point we can plot the model with matplotlib
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(obs_loglam, obs_flux, label = 'Obs.')
    ax2.plot(obs_loglam, obs_err, label = 'Obs. unc.')
    ax2.set_xlabel('Wavelength (A)')
    ax1.set_ylabel('Flux (arbitrary units)')
    ax2.set_ylabel('Uncertainty (arbitrary units)')
    fig.align_ylabels()
    
    # Now we are going to resample this model to be sampled linearly.We will employ
    # util module function fit_wave_interval (see miles_model_with_util.py example)
    # to obtain the wavelength arry of observation sampled linearly with the same
    # number of pixels.
    res_n_pixel = len(obs_loglam)
    res_lam = sc.util.fit_wave_interval(wave = obs_loglam,
                                        old_sampling = 'log',
                                        new_sampling = 'linear',
                                        new_size = res_n_pixel)
    
    # Using the resampling function with uncertainties. One need to pass as input
    # the original flux, the priginal wavelength array an its sample, the new
    # wavelength array and the new sampling type and the flux standard deviation.
    res_obs, res_wave, res_err = sc.resampling(flux = obs_flux,
                                               old_wave = obs_loglam,
                                               old_sampling_type = 'log',
                                               new_wave = res_lam,
                                               new_sampling_type = 'linear',
                                               flux_err = obs_err)
    
    # Finally plot resampled model
    ax1.plot(res_lam, res_obs, label = 'Resampled obs')
    ax2.plot(res_lam, res_err, label = 'Resampled obs. unc.')
    ax1.legend()
    ax2.legend(loc='best', bbox_to_anchor = (0.4, 0.5, 0.4, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_example_3()
