# tests/conftest.py
'''
Test-suite organisation for DINGUS.

Tests are tagged along TWO INDEPENDENT AXES, because "what kind of evidence is this?" and "how long
does it take?" are different questions and they will not stay correlated.

WHAT KIND OF EVIDENCE (pick exactly one):

    unit        Is this OPERATOR correct?  One piece of machinery, in isolation, against hand-computed
                algebra. Deterministic, no physics claim.
                e.g. the stress tensor is symmetric; the BR1 gradient of a linear field is exact;
                     the wall's central trace equals the wall state.

    numerics    Is the DISCRETIZATION correct?  Consistency, order of accuracy, convergence rates --
                the mathematics of the scheme, not the physics of the answer.
                e.g. the exact solution has ~zero residual and it falls SPECTRALLY with poly_deg;
                     a deliberately WRONG boundary condition must break that (negative controls).

    physics     Does it reproduce a KNOWN PHYSICAL RESULT?  The end-to-end claim.
                e.g. Taylor-Green decays at 4K^2/Re; Couette relaxes onto the exact linear profile.

    regression  Does a FIXED BUG stay fixed?  Not a statement about operators, schemes, or physics --
                a tripwire. Kept separate so it does not muddy the taxonomy above.
                e.g. the weakly-imposed Dirichlet wall is stable (it once diverged and blew up).

HOW EXPENSIVE (orthogonal):

    slow        Minutes of explicit time-marching. Requires --runslow. Everything else runs by default.

WHY THE TWO AXES ARE SEPARATE: today `physics` happens to be slow and `numerics` happens to be fast,
but that is a coincidence of the current cases, not a law. The Couette/Poiseuille NEGATIVE CONTROLS
are `numerics` and take milliseconds; a cheap physics check could exist tomorrow. Baking "slow" into
"physics" would force a re-tag the moment that correlation breaks.

USAGE
    pytest                              unit + numerics + regression(fast)   ~15 s   <- inner loop
    pytest --runslow                    EVERYTHING                           ~50 min <- before committing
    pytest -m unit                      operators only                       ~11 s
    pytest -m numerics                  the scheme only                      ~3 s
    pytest -m "physics" --runslow       the physical validation only         ~45 min
    pytest -m regression --runslow      the bug tripwires only               ~22 min
    pytest --markers                    print these descriptions

A bare `pytest` REPORTS the slow tests it skipped rather than pretending they do not exist -- which is
what makes it usable as the "does this fresh clone work?" smoke test.
'''
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run the slow tests too (minutes of explicit time-marching; ~50 min total)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: one operator, in isolation, vs hand-computed algebra (fast)")
    config.addinivalue_line("markers", "numerics: the discretization is consistent and converges (fast)")
    config.addinivalue_line("markers", "physics: reproduces a known physical result (usually slow)")
    config.addinivalue_line("markers", "regression: a tripwire for a bug that was fixed once already")
    config.addinivalue_line("markers", "slow: minutes of time-marching; needs --runslow")


def pytest_collection_modifyitems(config, items):
    '''Skip anything marked `slow` unless --runslow was passed. The skips are REPORTED, so a bare
    `pytest` tells you what it did not run instead of silently omitting it.'''
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="slow test; pass --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
