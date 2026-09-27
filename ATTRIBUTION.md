# Attribution and Redistribution

`bandit_lp.py` and `strategies.py` are reference implementations provided by
Nicolas Gast and are included in this public repository with his permission.
The remaining benchmark, instance, caching, learning, plotting, and reporting
modules are project extensions built around that interface.

The September 2026 project changes to the reference files fix integer action
accounting, FTVA zero-occupancy probabilities and virtual initialization, state
rounding, LP status checks, and simulation-cache behavior. These changes are
documented in `CORRECTIONS.md` and should be distinguished from the supplied
reference implementations. The original revision remains in Git history.

The inclusion permission does not by itself define a general open-source
license for the two reference files. Users planning redistribution beyond
research use should confirm the applicable terms with the authors.
