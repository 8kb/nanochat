"""
modelcore.tests -- modelcore's own test suite. Fully self-contained: nothing here imports
nanochat/scripts/tasks/dev (see test_standalone.py, which enforces this mechanically for the
package under test). GOLDENS_DIR is exported so nanochat's own tests (tests/test_architectures.py)
can cross-check against modelcore's tiny_composed_* goldens without duplicating them.
"""
import os

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "goldens")
TINY_DIR = os.path.join(GOLDENS_DIR, "tiny")
