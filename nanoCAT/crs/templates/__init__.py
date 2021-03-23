from __future__ import annotations

import sys
from types import MappingProxyType
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
import numpy as np
from scm.plams import Settings

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader  # type: ignore[misc]

if TYPE_CHECKING:
    if sys.version_info > (3, 8):
        from typing import Literal, TypedDict
    else:
        from typing_extensions import Literal, TypedDict

    _PresetKeys = Literal["MOPAC PM6", "ADF combi2005"]
    _JobKeys = Literal[
        "PUREVAPORPRESSURE", "VAPORPRESSURE", "PUREBOILINGPOINT", "BOILINGPOINT",
        "ACTIVITYCOEF", "LOGP",
    ]

    class _ParameterDict(TypedDict):
        description: str
        supports_multi_solute: bool
        supports_multi_solvent: bool
        properties: MappingProxyType[str, tuple[int, np.dtype[Any]]]
        default_properties: frozenset[str]
        settings: Settings

__all__ = ["PARAMETERS", "JOBS"]

_ROOT = Path(__file__).parent

with open(_ROOT / "presets.yaml", "r") as f:
    _PARAMETERS = yaml.load(f, Loader=SafeLoader)
    for k, v in _PARAMETERS.items():
        _PARAMETERS[k] = Settings(v)

with open(_ROOT / "jobs.yaml", "r") as f:
    _JOBS = yaml.load(f, Loader=SafeLoader)
    for k, v in _JOBS.items():
        v["settings"] = Settings(v["settings"])
        v["properties"] = MappingProxyType({k: (i, np.dtype(j)) for k, (i, j) in v["properties"]})
        v["default_properties"] = frozenset(v["default_properties"])

PARAMETERS: MappingProxyType[_PresetKeys, _ParameterDict] = MappingProxyType(_PARAMETERS)
JOBS: MappingProxyType[_JobKeys, Settings] = MappingProxyType(_JOBS)
