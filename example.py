import numpy as np
import pickle
from astropy.io import fits


model_wavelength = 10 * (10**(2.57671464 + np.arange(167283) * 2.25855074e-06)) # The extra 10* is for nm -> angstroms

with open("H.pkl", "rb") as fp:
    basis_vectors = pickle.load(fp)


def read_spectrum(path):
    with fits.open(path) as image:
        flux = image[0].data
        
        wl = np.exp(
            np.linspace(
                image[0].header["CRVAL1"],
                image[0].header["CDELT1"] * (len(flux) - 1) + image[0].header["CRVAL1"],
                len(flux)
            )
        )
    return (wl, flux)



from continuum import HERMESContinuumModel


paths = [
    ("data/01067079_HRF_OBJ_ext_CosmicsRemoved_log_merged_cf.fits", 70),
    ("data/01015922_HRF_OBJ_ext_CosmicsRemoved_log_merged_cf.fits", 107.5),
    ("data/00984687_HRF_OBJ_ext_CosmicsRemoved_log_merged_cf.fits", 200)
]

for path, v_rad in paths:
    wavelength, flux = read_spectrum(path)

    wavelength = wavelength * (1 + v_rad / 3e5)

    # Interpolate the data to model grid
    # Note: This is a BAD thing to do. For principled reasons you should always interpolate the model TO the data space
    #       because that is consistent with the idea of a "forward model" for the data. But here, the difference between
    #       interpolating the data -> model and model -> data will be very small, and it is computationally more
    #       efficient to have a consistent model grid.
    #       Also, we shouldn't be using a linear interpolation, but it will be good enough for continuum normalization purposes.

    interp_flux = np.interp(model_wavelength, wavelength, flux, left=np.nan, right=np.nan)

    # create an inverse variance array
    interp_ivar = np.ones_like(interp_flux)

    # set bad pixels to have zero inverse variance (zero inverse variance means 'no information')
    is_bad_pixel = (
        ~np.isfinite(interp_flux)
    |   (interp_flux <= 0)
    )
    interp_flux[is_bad_pixel] = 0
    interp_ivar[is_bad_pixel] = 0

    # This is just to exclude the biggest telluric bands. 
    # The place of these bands will vary per observation, but these are just selected to be wide enough to capture most of it.
    is_likely_telluric = (
        ((5850 < model_wavelength) & (model_wavelength < 6100))
    |   ((6250 < model_wavelength) & (model_wavelength < 6350))
    |   ((6860 < model_wavelength) & (model_wavelength < 7440))
    |   ((7590 < model_wavelength) & (model_wavelength < 7750))
    )
    interp_ivar[is_likely_telluric] = 0

    # restrict to 5000 Angstroms for now while we test

    # As a rule of thumb, you should set L to be 1x or 2x the width of the region you are fitting. So if your wavelength region
    # is from 3000 - 9000 angstroms and you are fitting that as 1 region, then you should set L to be 6000 or 12000.
    model = HERMESContinuumModel(
        deg=6,
        L=None,
        regions=[
            #[4000, 5000],
            #[6500, 6600]
            [3850, 8000] # Paschen jump not well modelled by the BOSZ grid.
        ]
    )

    continuum, meta = model.fit(interp_flux, interp_ivar, full_output=True)

    fig, (ax, ax_norm) = plt.subplots(2, 1)
    ax.plot(model_wavelength, interp_flux, c='k')
    ax.plot(model_wavelength, continuum[0], c="tab:blue")

    ax_norm.plot(model_wavelength, interp_flux / continuum[0], c="k")
    xlim = ax_norm.get_xlim()
    ax_norm.plot(model_wavelength, meta["rectified_model_flux"], c="tab:red")
    ax_norm.set_xlim(xlim) # restrict to just the region we fit
    ax_norm.set_ylim(0, 1.2)
    ax_norm.set_xlabel("wavelength [vac; A]")
    
    fig.savefig(path.replace(".fits", ".png"))
    