"""File entry point for FunASR import_module_from_path(remote_code)."""
import importlib.util
from pathlib import Path
import sys

_PACKAGE = "funasr_wav2vec_bert_adapter"
_ROOT = Path(__file__).resolve().parent
if _PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE, None)
        raise
