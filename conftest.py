"""Make the project importable when pytest is run from ANY working directory.

pytest auto-loads this conftest (it sits at the project root, which becomes
the rootdir), and inserts the project root into sys.path so `import mte` and
`import config` resolve regardless of the caller's cwd.  Run from anywhere:

    python -m pytest D:\\MarkovSoccerTE\\tests -x
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
