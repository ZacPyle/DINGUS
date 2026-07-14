# tests/SingleTests/testInputs/source_term_manufactured.py
from dingus.config import CaseCfg
import numpy as np

'''
Source-term fixture for test_source_term.py. Deliberately depends on BOTH the state q and the physical
coordinates (x, y), so the test can confirm that the residual hands source_term() the right state at the
right node -- a transposed or mis-mapped coordinate array would sail through a constant source.
'''

def source_term(case_config: CaseCfg, q, x, y, t) -> np.ndarray:
    # S = q * (1 + x + 2y + 3t) : state-dependent AND coordinate-dependent AND time-dependent.
    return q * (1.0 + x + 2.0 * y + 3.0 * t)[..., None]
