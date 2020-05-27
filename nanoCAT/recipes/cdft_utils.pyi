import sys
from typing import Optional, Mapping, Type, Any, overload, List
from scm.plams import Settings, Molecule, ADFJob, ADFResults
from qmflows import Settings as QmSettings
from qmflows.packages.SCM import ADF, ADF_Result

if sys.version_info < (3, 7):
    from typing_extensions import Literal
else:
    from typing import Literal

__all__: List[str] = ...

cdft: QmSettings = ...

@overload
def conceptual_dft(mol: Molecule, setting: Mapping, job_type: ADF = ..., template: Optional[Settings] = ..., **kwargs: Any) -> ADF_Result: ...
@overload
def conceptual_dft(mol: Molecule, setting: Mapping, job_type: Type[ADFJob] = ..., template: Optional[Settings] = ..., **kwargs: Any) -> ADFResults: ...

