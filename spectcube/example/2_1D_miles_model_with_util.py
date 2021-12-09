import spectcube as sc
import matplotlib.pyplot as plt

from astropy.io import fits
from os import path


def run_example_2():
    # Resampling MILES model using some auxiliary function of util module.
    
    # setting path to the module directory
    pack_dir = path.dirname(path.realpath(sc.__file__))
    
    # Path to model data
    model_path = pack_dir + '/model/Mun1.30Zm0.40T00.0631_iPp0.00_baseFe_linear_FWHM_2.51.fits'
    
    # Read model data
    hdu = fits.open(model_path)
    
    # The model flux array is then extracted with
    model_flux = hdu[0].data
    
    # Extract some informations from fits file to create an array
    # with the values of wavelength range. This model is sampled linearly,
    # this information can be obtained from the fits header.
    # For details about fits keywords see https://arxiv.org/abs/astro-ph/0507293
    wave_start = hdu[0].header['CRVAL1']
    wave_step = hdu[0].header['CDELT1']
    wave_n_pixel = hdu[0].header['NAXIS1']
    
    # In the 'util' module we provide some tools to facilitate the creation of
    # wavelength arrays.
    # util.build_wave_array return an array using as input the starting wavelength,
    # the steps beetween wavelengths, the number of points and
    # the sampling type ('linear', 'log' or 'ln').
    
    # Creating array with the original lambda wavelength values
    model_lam = sc.util.build_wave_array(wave = [wave_start, wave_step],
                                         sampling_type = 'linear',
                                         size = wave_n_pixel)
    
    # At this point we can plot the model with matplotlib
    plt.plot(model_lam, model_flux, label = 'Model')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (arbitrary units)')
    
    # Now we are going to resample this model to be sampled logarithmically.
    # For this we will create the new wavelenght array with the desired values.
    # In this example the log array will have the same starting point, end point
    # and number of point thus the step value is the unknown here. We will employ
    # another function of util module called fit_wave_interval whose input arguments
    # original wavelength array, original sampling type (linear, log, ln), and
    # the new wavelenght sampling sampling type ('log' here).    
    res_lam = sc.util.fit_wave_interval(wave = model_lam,
                                        old_sampling = 'linear',
                                        new_sampling = 'log')
    
    # Using the resampling function. One need to pass as input arguments the original flux,
    # the priginal wavelength array an its sample, the new wavelength array and
    # the new sampling type
    res_model, res_wave, _ = sc.resampling(flux = model_flux,
                                           old_wave = model_lam,
                                           old_sampling_type = 'linear',
                                           new_wave = res_lam,
                                           new_sampling_type = 'log')
    
    # Finally plot resampled model
    plt.plot(res_lam, res_model, label = 'Resampled model')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_example_2()