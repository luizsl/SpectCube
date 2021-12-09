import glob
import numpy as np
import spectcube as sc
import matplotlib.pyplot as plt

from astropy.io import fits
from os import path

def run_example_4():
    # Resampling multiple MILES models
    
    # setting path to the module directory
    pack_dir = path.dirname(path.realpath(sc.__file__))
    
    # Path to model data
    model_path = glob.glob(pack_dir + '/model/*.fits')
    
    
    #                           *** Note ***
    #
    # The model fluxes should always be along the axis 0. Here we have 3 models 
    # with 4300 pixel each. Thus the shape of the array is (4300, 3)
    # i.e. 3 columns and 4300 rows.
    #
    #                              * * *
    
    for index, file in enumerate(model_path):
        with fits.open(file) as hdu:
            # Create 2D empty array for the models
            model_len = len(hdu[0].data)
            model_n = len(model_path)
            model_flux = np.zeros(shape = (model_len, model_n))
    
            # ... and the array with the wavelegth values
            # we use the sc.util module but one can create it with numpy
            wave_start = hdu[0].header['CRVAL1']
            wave_step = hdu[0].header['CDELT1']
            wave_pixel = hdu[0].header['NAXIS1']
    
            model_lam = sc.util.build_wave_array(wave = [wave_start, wave_step],
                                                 sampling_type = 'linear',
                                                 size = model_len)
            break
    
    # Read model data and fill the empty array
    for index, file in enumerate(model_path):
        with fits.open(file) as hdu:
            model_flux[:, index] = hdu[0].data
    
    # Setting the new array the wavelengths to be the same,
    # just changing linear sampling to sampling ln.
    # Again we employ sc.util and again it is possible to choose another options to
    # build a simple numpy array
    res_lam = sc.util.fit_wave_interval(wave = model_lam,
                                        old_sampling = 'linear',
                                        new_sampling = 'ln',
                                        new_size = model_len)
    
    # Resampling
    res_model, res_wave, _ = sc.resampling(flux = model_flux,
                                           old_wave = model_lam,
                                           old_sampling_type = 'linear',
                                           new_wave = res_lam,
                                           new_sampling_type = 'ln')
    
    # Plot of models before and after resampling
    plt.plot(model_lam, model_flux[:, 0], label = 'MILES Model 1')
    plt.plot(model_lam, model_flux[:, 1], label = 'MILES Model 2')
    plt.plot(model_lam, model_flux[:, 2], label = 'MILES Model 3')
    
    plt.plot(res_wave, res_model[:, 0], label = 'Res. Model 1')
    plt.plot(res_wave, res_model[:, 1], label = 'Res. Model 2')
    plt.plot(res_wave, res_model[:, 2], label = 'Res. Model 3')
    
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (arbitrary units)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_example_4()
