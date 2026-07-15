# tests/UnitTests/test_viscosity_models.py
'''
Unit tests for the dynamic-viscosity models (constitutiveRelations.compute_viscosity):

  - 'constant'   : mu* = 1 everywhere (the default; MMS / Taylor-Green rely on it).
  - 'sutherland' : mu* = (T*)^(3/2) (1 + S*)/(T* + S*),  S* = 110.4 / ref_temperature.

The Sutherland tests pin the properties that make it correct and consistent:
  1. mu* = 1 EXACTLY at the reference state T* = 1 (so it agrees with the mu_ref in Re/Pr).
  2. mu* rises with temperature and falls below 1 for cold gas (the physical trend for air).
  3. a hand-computed value matches, for a specific T* and ref_temperature.
'''
import numpy as np
import pytest

from dingus.config import CaseCfg, PhysicsCfg
from dingus.physics.constitutiveRelations import compute_viscosity, compute_temperature


def _cfg(viscosity_model='constant', ref_temperature=None, sutherland_constant=None):
    physics = {'model': 'navier-stokes', 'Re': 100.0, 'Pr': 0.71, 'mach_ref': 1.0, 'gamma': 1.4,
               'riemann_solver': 'roe', 'viscosity_model': viscosity_model}
    if ref_temperature is not None:
        physics['ref_temperature'] = ref_temperature
    if sutherland_constant is not None:
        physics['sutherland_constant'] = sutherland_constant
    return CaseCfg.model_validate({
        'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': 'u.mesh', 'ndim': 2, 'poly_deg': 2, 'quad_type': 'LG'},
        'physics':        physics,
        'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
        'initialization': {'IC_method': 'analytical', 'IC_file': 'ic.py'},
        'io':             {'output_format': 'vtk', 'output_dir': './o/'},
    })


def _rest_state_at(T_star, cfg):
    '''A fluid-at-rest conserved state [rho, 0, 0, rhoE] whose nondimensional temperature is T_star.
    T* = gamma * mach_ref^2 * p / rho, so with rho = 1, p = T* / (gamma * mach_ref^2), and (u = 0) rhoE
    = p / (gamma - 1).'''
    gamma = cfg.physics.gamma
    p     = T_star / (gamma * cfg.physics.mach_ref**2)
    return np.array([[1.0, 0.0, 0.0, p / (gamma - 1.0)]])


def test_constant_model_is_unity():
    cfg = _cfg('constant')
    q   = _rest_state_at(3.7, cfg)                 # any temperature
    assert np.allclose(compute_viscosity(q, cfg), 1.0)


def test_sutherland_is_unity_at_the_reference_temperature():
    '''mu* must be EXACTLY 1 at T* = 1 -- otherwise it contradicts the mu_ref baked into Re and Pr.'''
    cfg = _cfg('sutherland', ref_temperature=288.15)
    q   = _rest_state_at(1.0, cfg)
    assert np.isclose(compute_temperature(q, cfg)[0], 1.0), "test setup: state should be at T* = 1"
    assert np.allclose(compute_viscosity(q, cfg), 1.0), "Sutherland mu* must be 1 at the reference T*=1"


def test_sutherland_increases_with_temperature():
    '''Hot gas is more viscous, cold gas less -- mu* monotic in T*, crossing 1 at the reference.'''
    cfg = _cfg('sutherland', ref_temperature=288.15)
    T_stars = np.array([0.5, 0.8, 1.0, 1.5, 2.0, 4.0])
    mus     = np.array([compute_viscosity(_rest_state_at(T, cfg), cfg)[0] for T in T_stars])

    assert np.all(np.diff(mus) > 0), f"Sutherland mu* must increase with T*: {mus}"
    assert mus[T_stars == 0.5][0] < 1.0, "cold gas (T* < 1) must be LESS viscous than reference"
    assert mus[T_stars == 2.0][0] > 1.0, "hot gas (T* > 1) must be MORE viscous than reference"


def test_sutherland_matches_hand_computation():
    '''Pin a specific value: at T* = 2, ref_temperature = 288.15 -> S* = 110.4/288.15.'''
    T_ref = 288.15
    cfg   = _cfg('sutherland', ref_temperature=T_ref)
    T     = 2.0
    S     = 110.4 / T_ref
    mu_expected = T**1.5 * (1.0 + S) / (T + S)
    assert np.isclose(compute_viscosity(_rest_state_at(T, cfg), cfg)[0], mu_expected)


def test_sutherland_requires_reference_temperature():
    '''Selecting sutherland without ref_temperature is a config error (S* would be undefined).'''
    with pytest.raises(ValueError, match='ref_temperature'):
        PhysicsCfg.model_validate({'model': 'navier-stokes', 'Re': 100.0, 'viscosity_model': 'sutherland'})


def test_sutherland_constant_defaults_to_air_and_is_configurable():
    '''The Sutherland temperature S defaults to the air value (110.4 K) but can be overridden.'''
    T, T_ref = 2.0, 288.15

    # default S = 110.4 (air) when not specified
    cfg_air = _cfg('sutherland', ref_temperature=T_ref)
    assert cfg_air.physics.sutherland_constant == 110.4
    mu_air  = T**1.5 * (1.0 + 110.4 / T_ref) / (T + 110.4 / T_ref)
    assert np.isclose(compute_viscosity(_rest_state_at(T, cfg_air), cfg_air)[0], mu_air)

    # a custom S (e.g. a different gas) changes S* and hence mu*
    S_custom = 240.0
    cfg_gas  = _cfg('sutherland', ref_temperature=T_ref, sutherland_constant=S_custom)
    mu_gas   = T**1.5 * (1.0 + S_custom / T_ref) / (T + S_custom / T_ref)
    assert np.isclose(compute_viscosity(_rest_state_at(T, cfg_gas), cfg_gas)[0], mu_gas)
    assert not np.isclose(mu_gas, mu_air), "a different Sutherland constant must give a different mu*"
