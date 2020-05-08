import sys
from os import PathLike
from enum import IntEnum
from typing import Any, MutableMapping, Optional, Type, Union, overload, ContextManager, Callable, Tuple, List

from noodles.interface import PromisedObject
from qmflows import Settings as QmSettings
from qmflows.packages import Package, Result
from scm.plams import Molecule, SingleJob, Settings, Results
from FOX import PSFContainer

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

__all__: List[str] = ...

@overload
def multi_ligand_opt(mol: Molecule, psf: Union[str, bytes, PathLike, PSFContainer], settings: MutableMapping, job_type: Type[SingleJob] = ..., path: Union[None, str, PathLike] = ..., folder: Union[None, str, PathLike] = ..., **kwargs: Any) -> Results: ...
@overload
def multi_ligand_opt(mol: Molecule, psf: Union[str, bytes, PathLike, PSFContainer], settings: MutableMapping, job_type: Package = ..., path: Union[None, str, PathLike] = ..., folder: Union[None, str, PathLike] = ..., **kwargs: Any) -> Result: ...

class _JobStyle(IntEnum):
    PLAMS: Literal[0] = ...
    QMFLOWS: Literal[1] = ...

@overload
def _parse_job(settings: MutableMapping, job_type: Type[SingleJob], path: Union[None, str, PathLike] = ..., folder: Union[None, str, PathLike] = ...) -> Tuple[Literal[0], ContextManager[None], Callable[[SingleJob], Results], Settings]: ...
@overload
def _parse_job(settings: MutableMapping, job_type: Package, path: Union[None, str, PathLike] = ..., folder: Union[None, str, PathLike] = ...) -> Tuple[Literal[1], ContextManager[None], Callable[[PromisedObject], Result], QmSettings]: ...
