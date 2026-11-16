"""Register the custom node folder as package `introsvg_node` for the tests.

The modules use package-relative imports; tests import them as
`introsvg_node.<module>` without a ComfyUI installation.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The decoder imports ComfyUI's `comfy` package from the installation holding this custom node folder.
if str(ROOT.parents[1]) not in sys.path:
    sys.path.append(str(ROOT.parents[1]))
if "introsvg_node" not in sys.modules:
    package = types.ModuleType("introsvg_node")
    package.__path__ = [str(ROOT)]
    sys.modules["introsvg_node"] = package
