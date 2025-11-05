# app/tests/conftest.py
from pathlib import Path
import sys

# ensure repo root is on sys.path so `import app` works when pytest runs from any cwd
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
