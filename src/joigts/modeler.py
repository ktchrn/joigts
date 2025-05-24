from collections.abc import Callable 
from typing import Tuple
from jax.typing import ArrayLike

import numpy as np
import jax
import jax.numpy as jnp
import numpyro as npr

from joigts.interface import Spectrum


def get_likelihood_gaussian(
        flux_model_synthesizer: Callable,
        spectrum: Spectrum) -> Callable:
    prefix = spectrum.name
    nflux_observed = np.array(spectrum.data['normed_flux'])
    nflux_stdev = np.array(spectrum.data['normed_flux_sd'])

    def likelihood_gaussian(*flux_model_args, **flux_model_kwargs):
        with npr.handlers.scope(prefix=prefix, divider='_'):
            flux_model = flux_model_synthesizer(
                *flux_model_args, **flux_model_kwargs
                )
            npr.deterministic('flux_model', flux_model)
            npr.sample(
                'flux_gaussian_likelihood', 
                npr.distributions.Normal(
                    loc=flux_model,
                    scale=nflux_stdev),
                obs=nflux_observed,
                )
    return likelihood_gaussian


def get_likelihood_gp(
        flux_model_func: Callable,
        spectrum: Spectrum,
        gp_amp=0.1,
        gp_scale=2.0) -> Callable:
    import tinygp

    @tinygp.helpers.dataclass
    class Multiplied(tinygp.kernels.quasisep.Wrapper):
        def coord_to_sortable(self, X):
            return X[0]

        def observation_model(self, X):
            return X[1] * self.kernel.observation_model(X[0])

    prefix = spectrum.name
    wave = np.array(spectrum.data['wavelength'].value)
    nflux_observed = np.array(spectrum.data['normed_flux'])
    nflux_stdev = np.array(spectrum.data['normed_flux_sd'])
    def likelihood_gp(*flux_model_args, **flux_model_kwargs):
        with npr.handlers.scope(prefix=prefix, divider='_'):
            plain_kernel = gp_amp**2 * tinygp.kernels.quasisep.Matern32(gp_scale)
            mult_kernel = Multiplied(plain_kernel)

            flux_model = flux_model_func(*flux_model_args, **flux_model_kwargs)
            npr.deterministic('flux_model', flux_model)

            gp = tinygp.GaussianProcess(
                kernel=mult_kernel,
                X=[wave, flux_model],
                mean=lambda X: X[1], diag=nflux_stdev**2)
            npr.sample(
                'likelihood', 
                gp.numpyro_dist(),
                obs=nflux_observed)
    return likelihood_gp
                

def get_voigt_flux_model_synthesizer(
        normed_flux_synthesizer: Callable,
        convolver: Callable,
        continuum: ArrayLike,
        comp_names: ArrayLike) -> Callable:
    comp_names = np.asarray(comp_names)
    continuum = np.asarray(continuum)
    def flux_model_synthesizer(log10_N_dict, vcen_km_s_dict, b_km_s_dict):
        param_arrays = _expand_parameter_dicts_to_arrays(
            log10_N_dict, vcen_km_s_dict, b_km_s_dict, comp_names
            )
        
        normed_flux = normed_flux_synthesizer(*param_arrays)
        conv_flux = convolver(normed_flux)

        return continuum*conv_flux

    return flux_model_synthesizer


def _expand_parameter_dicts_to_arrays(
        log10_N_dict: dict,
        b_km_s_dict: dict,
        vcen_km_s_dict: dict,
        comp_names: ArrayLike
        ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    log10_N_arr = jnp.array([log10_N_dict[key] for key in comp_names], dtype=float)
    b_km_s_arr = jnp.array([b_km_s_dict[key] for key in comp_names], dtype=float)
    vcen_km_s_arr = jnp.array([vcen_km_s_dict[key] for key in comp_names], dtype=float)

    return log10_N_arr, vcen_km_s_arr, b_km_s_arr
