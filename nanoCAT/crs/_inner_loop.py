from __future__ import annotations

import copy
import warnings
from itertools import islice
from typing import TYPE_CHECKING, TypeVar, Any
from collections.abc import Sequence, Mapping, Iterable

import numpy as np
from scm.plams import Settings, CRSJob
from more_itertools import chunked

from .templates import JOBS

if TYPE_CHECKING:
    from .templates import _JobKeys

    SCT = TypeVar("SCT", bound=np.generic)
    NDArray = np.ndarray[Any, np.dtype[SCT]]

__all__ = ["CRSExtractor"]


class CRSRunner:
    """Placeholder."""

    def __init__(
        self,
        solutes: str | None | Sequence[str | None] | NDArray[np.str_] | NDArray[np.object_],
        jobs: Mapping[_JobKeys, Iterable[str]],
    ) -> None:
        self.solutes: NDArray[np.object_] = np.array(solutes, copy=False, ndmin=1, dtype=np.object_)
        if self.solutes.ndim != 1:
            raise ValueError
        self.mask: NDArray[np.bool_] = self.solutes != None
        self.count: int = np.count_nonzero(self.mask)

    def run(
        self,
        job: _JobKeys,
        properties: Sequence[str],
        settings: None | Settings = None,
        solvents: None | Settings | list[Settings] = None,
        n: int = 1,
    ) -> NDArray[np.void]:
        """Run.

        Parameters
        ----------
        job : :class:`str`
            The type of CRS job.
        properties : :class:`Iterable[str] <collections.abc.Iterable>`
            The to-be returned properties.
        settings : :class:`plams.Settings <scm.plams.core.settings.Settings>`, optional
            Optional user-specified settings for updating the default templates.
        solvents : :class:`list[Settings] <list>`, optional
            Optional lists of settings with solvents.
            If a pure (*i.e.* unary) solvent is used one can just provide the embedded settings.

        Returns
        -------
        :class:`np.ndarray[np.void] <numpy.ndarray>`, shape :math:`(n_{solute},)`
            A 1D strucured array with COSMO-RS properties, the latter being used as field names.

        """
        # Create the to-be returned structured array
        ret = self._get_ret_array(job, properties)
        if self.count == 0:
            return ret

        # Property-specific offsets
        i = 0 if solvents is None else len(solvents)
        j = JOBS[job]["properties"] * n

        # Run the CRS Job
        s = self._construct_settings(job, solvents, settings)
        crsjob = CRSJob(settings=s)
        results = crsjob.run()
        if crsjob.status in ('failed', 'crashed'):
            return ret

        # Extract the properties
        field_dtype_iter = (tup[0] for tup in ret.dtype.fields.values())
        for field_dtype, prop in zip(field_dtype_iter, properties):
            # Extract the properties
            try:
                value = results.readkf(job, prop)
            except Exception as ex:
                msg = f"Failed to extract the {prop!r} property"
                warn = RuntimeWarning(msg)
                warn.__cause__ = ex
                warnings.warn(warn)

            # Cast the relevant property into an appropiate array
            else:
                if j == 1:
                    iterator: Iterable = (tuple(k) for k in chunked(islice(value, i * j, None), j))
                else:
                    iterator = value
                ret[prop][self.mask] = np.fromiter(iterator, dtype=field_dtype, count=self.count)
        return ret

    @staticmethod
    def _construct_settings(
        job: _JobKeys,
        solvents: None | Settings | list[Settings] = None,
        settings: None | Settings = None,
    ) -> Settings:
        """Construct job settings appropiate for the passed **job** and **solvents**.

        The to-be returned settings are updated with (optionally) user-specified **settings**.

        """
        s = copy.deepcopy(JOBS[job]["settings"])
        if solvents is None:
            s.input.compound = []
        elif isinstance(solvents, Settings):
            s.input.compound = [copy.deepcopy(solvents)]
        else:
            s.input.compound = copy.deepcopy(solvents)

        s.input.compound += []
        if settings is not None:
            s.update(settings)
        return s

    def _get_ret_array(self, job: _JobKeys, properties: Iterable[str]) -> NDArray[np.void]:
        """Construct a structured array appropiate for the passed **job** and **properties**."""
        # Construct the dtype
        dtype: list[tuple[str, np.dtype] | tuple[str, np.dtype, int]] = []
        for k in properties:
            i, field_dtype = JOBS[job]["properties"][k]
            if i != 1:
                dtype.append((k, field_dtype, i))
            else:
                dtype.append((k, field_dtype))

        # Construct the to-be returned array
        return np.zeros(len(self.solutes), dtype=dtype)
