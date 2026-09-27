"""Build the Kaggle kernel script: environment overrides followed by ``airbus_modern.py``.

Usage: python build_kernel.py [KEY=VALUE ...]   e.g.  python build_kernel.py ASD_TRAIN_HOURS=0.2
Then:  kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
"""

import sys
from pathlib import Path

here = Path(__file__).parent
overrides = dict(arg.split("=", 1) for arg in sys.argv[1:])
header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in overrides.items())
(here / "kernel" / "airbus_modern_kernel.py").write_text(header + "\n" + (here / "airbus_modern.py").read_text())
print(f"wrote kernel/airbus_modern_kernel.py with {overrides or 'default settings'}")
