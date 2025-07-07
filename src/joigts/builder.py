from collections.abc import Callable

import astropy.constants as aco
import astropy.units as u
from astropy.table import QTable
from jax.scipy.signal import convolve
import jax.numpy as jnp
import numpy as np

from jax_voigt.multiline import opacity_conversion_constant
from jax_voigt import astro_voigt_profile

from joigts import interface


def get_normed_voigt_flux_synthesizer(
        component_group_lines: QTable,
        eval_wave: u.Quantity,
        factor=1.0) -> Callable:
    """
    Get a function for a given component group and 
    evaluation wavelength array that takes column densities,
    centroid velocities, and broadening parameters and returns
    transmittances (aka normalized fluxes).

    Parameters
    ----------
    component_group_lines : QTable
        QTable of component group lines
    eval_wave : astropy.QTable.QTable
        QTable of spectrum chunk

    Returns
    -------
    """
    c_in_km_s = aco.c.to('km/s').value
    Γ_ν0 = (component_group_lines['gamma'] * component_group_lines['wrest'] / aco.c).to('').value

    z0 = ((eval_wave[:, None] /
          component_group_lines['wrest']).to('').value /
          (1+component_group_lines['zcen']) - 1)
    
    norm_factor = (
        component_group_lines['wrest'].to('cm').value *
        component_group_lines['f'].value *
        opacity_conversion_constant)

    def synthesize(log10_N, vcen_km_s, b_km_s):
        return jnp.exp((
            -1*(10**log10_N *
            norm_factor) *
            astro_voigt_profile(
                centroid_redshift=vcen_km_s/c_in_km_s,
                b_c=b_km_s/c_in_km_s,
                Γ_ν0=Γ_ν0,
                eval_redshift=z0,
            )
        ).sum(axis=-1))

    return synthesize


def get_windowed_normed_voigt_flux_synthesizer(
        component_group_lines: QTable,
        eval_wave: u.Quantity,
        factor=1.0,
        synth_vpad=np.inf*u.km/u.s) -> Callable:
    """
    Get a function for a given component group and 
    evaluation wavelength array that takes column densities,
    centroid velocities, and broadening parameters and returns
    transmittances (aka normalized fluxes).

    Parameters
    ----------
    component_group_lines : QTable
        QTable of component group lines
    eval_wave : astropy.QTable.QTable
        QTable of spectrum chunk

    Returns
    -------
    """
    c_in_km_s = aco.c.to('km/s').value
    Γ_ν0 = (component_group_lines['gamma'] * component_group_lines['wrest'] / aco.c).to('').value

    z0 = ((eval_wave[:, None] /
          component_group_lines['wrest']).to('').value /
          (1+component_group_lines['zcen']) - 1)
    tau_shape = z0.shape
    
    synth_wmin = component_group_lines['wmin'] * (1 - (synth_vpad/aco.c).to(''))
    synth_wmax = component_group_lines['wmax'] * (1 + (synth_vpad/aco.c).to(''))
    active_synth_mask = (
        (synth_wmin <= eval_wave[:, None]) & (eval_wave[:, None] <= synth_wmax)
    )
    wave_inds, line_inds = np.where(active_synth_mask)
    z0_active = z0[wave_inds, line_inds]

    norm_factor = (
        component_group_lines['wrest'].to('cm').value *
        component_group_lines['f'].value *
        opacity_conversion_constant)

    def synthesize(log10_N, vcen_km_s, b_km_s):
        tau = jnp.zeros(tau_shape)
        tau_flat = ((10**log10_N[line_inds] *
            norm_factor[line_inds]) *
            astro_voigt_profile(
                centroid_redshift=(vcen_km_s/c_in_km_s)[line_inds],
                b_c=(b_km_s/c_in_km_s)[line_inds],
                Γ_ν0=Γ_ν0[line_inds],
                eval_redshift=z0_active,
            )
        )
        tau = jnp.place(tau, active_synth_mask, tau_flat, inplace=False).sum(axis=1)
        return jnp.exp(-tau)

    return synthesize


def get_convolved_spectrum_func(
        lsf_func: Callable,
        eval_wave: u.Quantity,
        n_subdivide=1) -> Callable:
    lsf_array = np.array(interface.get_lsf(lsf_func, eval_wave))

    def convolved_spectrum_func(normalized_flux_array):
        conv_flux = (
            1 +
            convolve(normalized_flux_array-1, lsf_array, mode='same')
            )
        return conv_flux[::n_subdivide]

    return convolved_spectrum_func
    