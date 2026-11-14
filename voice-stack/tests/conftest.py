import os
import sys

# The servers are flat modules in voice-stack/ (no package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Some test modules stub `torch` when it isn't imported yet; import the real
# one first (when installed) so a stub never shadows it for the whole session.
try:
    import torch  # noqa: F401
except ImportError:
    pass
