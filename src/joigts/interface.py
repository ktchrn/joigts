from collections.abc import Callable
from typing import Tuple, List

import numpy as np
from astropy.table import QTable, Column, vstack
import astropy.constants as aco
import astropy.units as u
from linetools.spectra.lsf import LSF
from dataclasses import dataclass


_LINELISTS = {}


@dataclass
class Spectrum:
    name: str
    data: QTable
    lsf_config: dict

    def get_subrange(self, left_index, right_index, name=None):
        if name is None:
            name = self.name
        return Spectrum(
            name=name,
            data=self.data[left_index:right_index],
            lsf_config=self.lsf_config)


# getting line data for component defs
def parse_components(component_dict: dict) -> QTable:
    compname_list = []
    species_list = []
    line_name_list = []
    wrest_list = []
    f_list = []
    gamma_list = []
    zmin_list = []
    zmax_list = []
    for comp_name, comp in component_dict.items():
        if 'lines' not in comp:
            lines = get_lines_for_species(
                comp['species'],
                comp['linelist']['source'],
                comp['linelist']['name']
                )
            n_lines = len(lines)
        else:
            lines = comp['lines']
            for col in ['wrest', 'gamma']:
                # inner + outer Quantity to force conversion to single unit
                lines[col] = Column(
                    u.Quantity([u.Quantity(x) for x in lines[col]])
                    )
            n_lines = len(lines['wrest'])

        compname_list.extend([comp_name] * n_lines)
        species_list.extend([comp['species']] * n_lines)
        line_name_list.extend(list(lines['line_name']))
        wrest_list.extend([w * lines['wrest'].unit for w in lines['wrest']])
        f_list.extend(list(lines['f']))
        gamma_list.extend([g * lines['gamma'].unit for g in lines['gamma']])
        
        # also get z ranges to calculate wmin, wmax
        z_min = (
            (1+comp['zcen']) *
            (1+(comp['vmin']/aco.c).to('')) - 1)
        z_max = (
            (1+comp['zcen']) *
            (1+(comp['vmax']/aco.c).to('')) - 1)
        zmin_list.extend([z_min] * n_lines)
        zmax_list.extend([z_max] * n_lines)
    
    component_lines = QTable(data={
        'compname': compname_list,
        'species': species_list,
        'line_name': line_name_list,
        'wrest': wrest_list,
        'f': f_list,
        'gamma': gamma_list,
    })

    component_lines['wmin'] = component_lines['wrest'] * (1+np.array(zmin_list))
    component_lines['wmax'] = component_lines['wrest'] * (1+np.array(zmax_list))
    component_lines['zcen'] = np.array([
        component_dict[comp_name]['zcen']
        for comp_name in compname_list])

    return component_lines


def get_lines_for_species(
        species: str, 
        linelist_source: str,
        linelist_name: str) -> QTable:
    linelist = get_linelist(linelist_source, linelist_name)
    matches = linelist['species'] == species
    return linelist[matches]


# matching possible component lines to available spectra
def match_component_lines_and_spectrum(
        component_lines: QTable,
        spectra: List[Spectrum],
        vpad: u.Quantity = 50.0*u.km/u.s,
        check_all_present=True) -> Tuple[List[QTable], List[Spectrum]]:
    # copying because we'll be changing wmin and wmax
    component_lines = component_lines.copy()
    comp_names = component_lines['compname'].value

    zp1_blue, zp1_red = (1-(vpad/aco.c).to('')), (1+(vpad/aco.c).to(''))
    component_lines['wmin'] = component_lines['wmin'] * zp1_blue
    component_lines['wmax'] = component_lines['wmax'] * zp1_red

    component_line_groups = []
    all_spectrum_chunks = []

    for spectrum in spectra:
        spec_lines = _filter_lines_to_spec_range(component_lines, spectrum)

        spec_line_groups = _group_component_lines(spec_lines)
        spec_lines_groups, spectrum_chunks = _split_spectrum_to_chunks(
            spec_line_groups, spectrum)
        component_line_groups.extend(spec_line_groups)
        all_spectrum_chunks.extend(spectrum_chunks)
    
    if check_all_present:
        # in case there are no matches at all
        if len(component_line_groups) > 0:
            active_comps = np.concatenate(
                [g['compname'].value for g in component_line_groups])
        else:
            active_comps = []
        
        if not np.in1d(comp_names, active_comps).all():
            missing_comps = np.setdiff1d(comp_names, active_comps)
            raise ValueError(
                f'Components without lines in spectrum: {', '.join(missing_comps)}')

    return component_line_groups, all_spectrum_chunks


def _filter_lines_to_spec_range(
        component_lines: QTable, 
        spectrum: Spectrum) -> QTable:
    # alert alert: assuming spectrum is increasing in wavelength
    spec_wmin, spec_wmax = spectrum.data[0]['wavelength'], spectrum.data[-1]['wavelength']
    outside_range = (
        (component_lines['wmax'] < spec_wmin) |
        (spec_wmax < component_lines['wmin'])
        )
    return component_lines[~outside_range]


def _group_component_lines(component_lines: QTable) -> Tuple[QTable]:
    if len(component_lines) == 0:
        return ()

    component_lines.sort(keys='wmin')
    wmins = component_lines['wmin']
    wmaxs = component_lines['wmax']

    group_idxs = []
    current_group_idxs = [0]
    left = wmins[0]
    group_right = wmaxs[0]
    active_group = True
    for idx in range(1, len(component_lines)):
        right = wmaxs[idx]
        if wmins[idx] <= group_right:
            # case: new left still in group
            current_group_idxs.append(idx)
            if group_right < right:
                # if this line's wmax would extend the group range, 
                # then it's the group's new right boundary
                group_right = right
        else:
            # case: new group
            group_idxs.append(current_group_idxs)
            current_group_idxs = [idx]
            group_right = wmaxs[idx]

    # append last group
    group_idxs.append(current_group_idxs)
    component_line_groups = tuple(component_lines[idxs] for idxs in group_idxs)

    return component_line_groups


def _split_spectrum_to_chunks(
        component_line_groups: List[QTable],
        spectrum: Spectrum) -> Tuple[List[QTable], List[Spectrum]]:
    spectrum_chunks = []
    data = spectrum.data
    
    chunk_idx = 0
    active_component_line_groups = []

    for group in component_line_groups:
        wmin = (
            group['wmin'].min() 
            * group['wmin'].unit.to(data['wavelength'].unit)
            )
        wmax = (
            group['wmax'].max() 
            * group['wmax'].unit.to(data['wavelength'].unit)
            )
        left_idx, right_idx = np.searchsorted(
            data['wavelength'], [wmin, wmax]
            )
        left_idx = max(0, left_idx-1)
        spectrum_chunk = spectrum.get_subrange(
            left_idx, right_idx, name=f'{spectrum.name}__chunk_{chunk_idx}')
        
        if len(spectrum_chunk.data) < 2:
            # for example if group falls in gap in spectrum
            continue

        chunk_idx += 1
        group = _filter_lines_to_spec_range(group, spectrum_chunk)
        active_component_line_groups.append(group)
        spectrum_chunks.append(spectrum_chunk)
    return active_component_line_groups, spectrum_chunks


# linelist management
def get_linelist(linelist_source: str, linelist_name: str) -> QTable:
    if linelist_source == 'linetools':
        return _get_linetools_linelist(linelist_name)
    else:
        raise ValueError(f"Unrecognized `linelist_source`: {linelist_source}.")


def _get_linetools_linelist(linelist_name):
    key = ('linetools', linelist_name)
    if key not in _LINELISTS:
        from linetools.lists.linelist import LineList
        ll = LineList(linelist_name)._data
        ll['species'] = [n.split(' ')[0] for n in ll['name']]
        ll['name'].info.name = 'line_name'
        _LINELISTS[key] = ll
    return _LINELISTS[key]


# other spectrum utilities
def upsample_wavelength(
        wavelength: u.Quantity,
        velocity_step: u.Quantity) -> Tuple[QTable, int]:
    """Create an upsampled wavelength axis with sampling better than the specified velocity step.
    
    Parameters
    ----------
    wavelength : astropy.QTable.QTable
        QTable containing a 'wavelength' column with Astropy units
    velocity_step : astropy.units.Quantity
        Maximum desired velocity step between wavelength points
        
    Returns
    -------
    astropy.units.Quantity
        New wavelength array with sampling better than velocity_step
    int
        Number of subdivisions used to achieve the new sampling
    """    
    # Calculate velocity steps between current points
    wave = wavelength
    dv = aco.c.to(velocity_step.unit) * (np.diff(wave) / wave[:-1]).to('')
    
    # Find median current velocity step
    med_dv = np.median(dv)
    
    # Calculate required number of subdivisions
    n_subdivide = int(np.ceil(med_dv / velocity_step))
    
    # Create array of subdivision points
    t = np.linspace(0, 1, n_subdivide, endpoint=False)
    
    # Use broadcasting to interpolate between all pairs of points at once
    new_wave = (
        wave[:-1, np.newaxis] *
        (1-t) + wave[1:, np.newaxis] * t).ravel()
    
    # Add final wavelength point
    new_wave = np.append(new_wave, wave[-1])
    
    return new_wave, n_subdivide


def get_lsf(lsf_func: Callable, wavelength: u.Quantity) -> np.ndarray:
    """Get LSF kernel values.
    
    Parameters
    ----------
    lsf_func : Callable
        Function that takes unitful wavelength arrays and returns the 
        line spread function relative to the wavelength array center
    wavelength : array-like
        Wavelength array to compute LSF for
        
    Returns
    -------
    lsf : array-like 
        LSF kernel values
    """
    # first evaluate LSF over full wavelength array
    lsf = lsf_func(wavelength)
    # for long wavelength arrays, lsf will mostly be zero
    positive_mask = lsf > 0
    sub_wave = wavelength[positive_mask]
    # re-evaluate lsf to ensure correct centering
    lsf = lsf_func(sub_wave)
    return lsf
