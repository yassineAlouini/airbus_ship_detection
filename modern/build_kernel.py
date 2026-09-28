"""Build a single-file Kaggle kernel script: environment overrides + the pipeline code.

Stage 1 (U-Net):       python build_kernel.py [KEY=VALUE ...]
                       kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
Stage 2 (ViT gate):    python build_kernel.py --stage vit [KEY=VALUE ...]
                       kaggle kernels push -p kernel_vit --accelerator NvidiaTeslaT4
Stage 3 (diagnosis):   python build_kernel.py --stage diag [KEY=VALUE ...]
                       kaggle kernels push -p kernel_diag --accelerator NvidiaTeslaT4

A Kaggle script kernel is a single file, so later stages concatenate the modules they depend on, dropping the
sibling imports and every ``__main__`` block except the last one.
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

STAGES = {
    "unet": (["airbus_modern.py"], "kernel/airbus_modern_kernel.py"),
    "vit": (["airbus_modern.py", "vit_gate.py"], "kernel_vit/vit_gate_kernel.py"),
    "diag": (["airbus_modern.py", "vit_gate.py", "diagnose.py"], "kernel_diag/diagnose_kernel.py"),
}
if stage not in STAGES:
    sys.exit(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
files, out = STAGES[stage]
parts = []
for i, name in enumerate(files):
    code = (here / name).read_text()
    # Sibling-module imports are satisfied by the concatenation itself.
    code = re.sub(r"from (airbus_modern|vit_gate) import (\([^)]*\)|[^\n]*)\n", "", code)
    if i < len(files) - 1:
        code = code.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")
    parts.append(code)
out = here / out
out.write_text(header + "\n" + "\n\n".join(parts))
print(f"wrote {out.relative_to(here)} with {overrides or 'default settings'}")
