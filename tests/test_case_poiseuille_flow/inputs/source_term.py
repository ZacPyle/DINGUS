# tests/test_case_poiseuille_flow/inputs/source_term.py
from dingus.config import CaseCfg
import numpy as np

'''
The body force that DRIVES Poiseuille flow.

A channel that is periodic in x has no mean pressure drop -- the pressure is the same at the inlet and
the outlet by construction -- so there is nothing to push the fluid along. The standard fix is to
replace the missing pressure gradient with an equivalent constant body force:

        -dp/dx  ->  G   (a constant force per unit volume, acting in +x)

which enters the conservation law as a source term  q_t + div(F) = S  with

        S = [ 0 ,  G ,  0 ,  G * u ]
              ^    ^    ^      ^
           mass  x-mom y-mom  energy

The ENERGY component is easy to forget and essential: a force acting on a moving fluid does WORK at a
rate (force . velocity) = G * u, and that work has to enter the energy budget. Drop it and the channel
quietly comes out at the wrong temperature.

It is also why source_term() is handed the state q: unlike a manufactured (MMS) source, which is a pure
function of (x, t), a body force generally depends on the SOLUTION -- here through u = (rho u) / rho.
'''

# Driving force (equivalent to a constant favourable pressure gradient -dp/dx = G). Chosen together
# with Re and H so that the centreline velocity u_max = G Re H^2 / 8 comes out to 1.0.
G = 0.8


def source_term(case_config: CaseCfg, q, x, y, t) -> np.ndarray:
    '''
    Inputs:
    - case_config : validated case configuration.
    - q           : (..., num_eq) conserved state at the quadrature nodes.
    - x, y        : (...,) physical coordinates of those nodes (unused: the force is uniform).
    - t           : current time (unused: the force is steady).

    Outputs:
    - S : (..., num_eq) source, one value per equation per node.
    '''
    rho = q[..., 0]
    u   = q[..., 1] / rho          # streamwise velocity

    S          = np.zeros_like(q)
    S[..., 1]  = G                 # x-momentum: the driving force itself
    S[..., 3]  = G * u             # energy: the RATE OF WORK done by that force
    return S
