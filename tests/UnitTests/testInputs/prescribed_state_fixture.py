import numpy as np
def boundary_state(cfg, x, y, t):
    rho = 1.0 + 0.1 * x + 0.2 * y
    u   = 0.3 * y
    v   = 0.1 * x
    p   = 2.0 + 0.05 * x
    g   = cfg.physics.gamma
    E   = p / (g - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E], axis=-1)
