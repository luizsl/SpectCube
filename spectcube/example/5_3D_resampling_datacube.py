import spectcube as sc
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from os import path


def run_example_5():
    # Resampling datacube. Some IFU data is stored in datacube
    # i.e. a 3D array of spectrums. In the example we use the same SDSS spectrum
    # four times to emulate a cube.
    
    
    #                           *** Note ***
    #
    # The model fluxes should always be along the axis 0. Here we emulate 4 models
    # in  a cube with 3841 pixel each. Thus the shape of the array is (3841, 2, 2)
    #
    #                              * * *
    
    
    # setting path to the module directory
    pack_dir = path.dirname(path.realpath(sc.__file__))
    
    # Path to observation data
    obs_path = pack_dir + '/data/spec-1665-52976-0519.fits'
    
    # Read observation data
    with fits.open(obs_path) as hdu:
        obs_flux = hdu['COADD'].data['flux']
        obs_err = (1.0/hdu['COADD'].data['ivar'])**0.5
        obs_loglam = 10**hdu['COADD'].data['loglam']
    
    # Emulate a datacube with SDSS spectrum adding a different constant
    # to each spectrum
    cube_flux = np.tile(obs_flux.reshape((-1,1,1)), (1,2,2)) + np.array([[0, 30],
                                                                         [60, 90]])
    cube_err = np.tile(obs_err.reshape((-1,1,1)), (1,2,2))

    res_lam = sc.util.fit_wave_interval(wave = obs_loglam,
                                        old_sampling = 'log',
                                        new_sampling = 'linear',
                                        new_size = len(obs_loglam))
    
    # Resampling
    res_cube, res_wave, res_err = sc.resampling(flux = cube_flux,
                                                old_wave = obs_loglam,
                                                old_sampling_type = 'log',
                                                new_wave = res_lam,
                                                new_sampling_type = 'linear',
                                                flux_err = cube_err)
    
    # Plot of models before and after resampling
    plt.plot(obs_loglam, cube_flux[:, 0 , 0], label = 'Cube spec 1')
    plt.plot(obs_loglam, cube_flux[:, 0 , 1], label = 'Cube spec 2')
    plt.plot(obs_loglam, cube_flux[:, 1 , 0], label = 'Cube spec 3')
    plt.plot(obs_loglam, cube_flux[:, 1 , 1], label = 'Cube spec 4')
    
    plt.plot(res_wave, res_cube[:, 0 , 0], label = 'Res spec 1')
    plt.plot(res_wave, res_cube[:, 0 , 1], label = 'Res spec 2')
    plt.plot(res_wave, res_cube[:, 1 , 0], label = 'Res spec 3')
    plt.plot(res_wave, res_cube[:, 1 , 1], label = 'Res spec 4')
    
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (arbitrary units)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_example_5()
