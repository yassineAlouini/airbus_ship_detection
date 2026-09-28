"""Build a single-file Kaggle kernel script: environment overrides + the pipeline code.

Stage 1 (U-Net):       python build_kernel.py [KEY=VALUE ...]
                       kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
Stage 2 (ViT gate):    python build_kernel.py --stage vit [KEY=VALUE ...]
                       kaggle kernels push -p kernel_vit --accelerator NvidiaTeslaT4

Stage 2 bundles ``airbus_modern.py`` (without its ``__main__`` block) followed by ``vit_gate.py`` (without its import
of ``airbus_modern``), because a Kaggle script kernel is a single file.
"""

import re
import sys
from pathlib import Path

here = Path(__file__).parent
args = sys.argv[1:]
stage = "unet"
if args[:1] == ["--stage"]:
    stage, args = args[1], args[2:]
overrides = dict(arg.split("=", 1) for arg in args)
header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in overrides.items())

base = (here / "airbus_modern.py").read_text()
if stage == "unet":
    body, out = base, here / "kernel" / "airbus_modern_kernel.py"
elif stage == "vit":
    base = base.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")
    vit = re.sub(r"from airbus_modern import \([^)]*\)\n", "", (here / "vit_gate.py").read_text())
    body, out = base + "\n\n" + vit, here / "kernel_vit" / "vit_gate_kernel.py"
else:
    sys.exit(f"unknown stage {stage!r}")
out.write_text(header + "\n" + body)
print(f"wrote {out.relative_to(here)} with {overrides or 'default settings'}")
