import numpy as np
import jax.numpy as jnp


# sizing: different for each kernel, static 
def matern_52_basis_size(length_scale, full_domain_half_width):
    return np.ceil(2.65 * (full_domain_half_width/length_scale)).astype(int)


def matern_52_full_domain_half_width(length_scale, active_domain_half_width):
    length_scale_crit = 4.1*length_scale/active_domain_half_width
    full_domain_half_width = (
        active_domain_half_width * np.where(length_scale_crit > 1.2, length_scale_crit, 1.2)
        )
    return full_domain_half_width


# eigenfunctions: same for all kernels, static (numpy)
def eigenfunctions(x, full_domain_half_width, basis_size):
    L = full_domain_half_width
    M = basis_size

    m1 = (np.pi / (2 * L)) * np.tile(L + x[:, None], M)
    m2 = np.diag(np.linspace(1, M, num=M))
    num = np.sin(m1 @ m2)
    den = np.sqrt(L)
    return num / den


# spectral densities: different for each kernel, partially dynamic
def get_matern_52_spectral_density_func(full_domain_half_width, basis_size):
    sqrt_eigenvalues = (
        np.arange(1, 1 + basis_size) * np.pi / 2 / full_domain_half_width
        )
    sqrt_2pi = np.sqrt(2*np.pi)
    def matern_52_spectral_density(amplitude, length_scale):
        c = amplitude * sqrt_2pi * length_scale
        e = jnp.exp(-0.5 * (length_scale ** 2) * (sqrt_eigenvalues ** 2))
        return c * e
    return matern_52_spectral_density


# convenience
def center_x(x_orig):
    """
    rescale an x vector to be centered
    """
    center = 0.5*(x_orig[0] + x_orig[-1])
    return x_orig - center
