import numpy as np
import matplotlib.pyplot as plt
import spectcube as sc

from os import path
from astropy.io import fits

def run_example_1():
    # In this example we do a simple resampling procedure with MILES model.
    # Some harcoding is used for sake of clarity.
    
    # setting path to the module directory
    pack_dir = path.dirname(path.realpath(sc.__file__))
    
    # Path to model data
    model_path = pack_dir + '/model/Mun1.30Zm0.40T00.0631_iPp0.00_baseFe_linear_FWHM_2.51.fits'
    
    # Read model data with astropy
    hdu = fits.open(model_path)
    
    # The model flux array is then extracted with
    model_flux = hdu[0].data
    
    # Extract some informations from fits file to create an array
    # with the values of wavelength range. This model is sampled linearly,
    # this information can be obtained from the fits header.
    # For details about fits keywords see https://arxiv.org/abs/astro-ph/0507293
    
    # First wavelegth value
    wave_start = hdu[0].header['CRVAL1']
    
    # Steps between coordinates
    wave_step = hdu[0].header['CDELT1']
    
    # Number of pixels
    wave_pixel = hdu[0].header['NAXIS1']
    
    # Creating array with the original lambda wavelength values. Here the starting
    # wavelength is 3540.5 A, the step between each wavelegth is 0.9 A with 4300
    # pixels thus going up to 7409.6 A. Now we create an array with these values
    # sampled linearly
    model_lam = 3540.5 + 0.9*np.arange(0, 4300, 1)
    
    # At this point we can plot the model with matplotlib
    plt.plot(model_lam, model_flux, label = 'Model')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (arbitrary units)')
    
    # Now we are going to resample this model in new wavelength points.  For this
    # we will create the new wavelength array with the desired values. In this
    # example the array will start at 3541 A and end at 7409 A with 1 A spacing.
    res_lam = 3541 + np.arange(0, 3869, 1)
    
    # Using the resampling function. One need to pass as input the original flux,
    # the priginal wavelength array an its sample, the new wavelength array and
    # the new sampling type
    res_model, res_wav, _ = sc.resampling(flux = model_flux,
                                          old_wave = model_lam,
                                          old_sampling_type = 'linear',
                                          new_wave = res_lam,
                                          new_sampling_type = 'linear')
    
    # Finally plot resampled model
    plt.plot(res_lam, res_model, label = 'Resampled model')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    run_example_1()
    