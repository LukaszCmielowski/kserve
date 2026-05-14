# Copyright 2026 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from kserve import Model
from kserve.errors import InferenceError, ModelMissingError
from kserve.logging import logger
from kserve.protocol.infer_type import InferRequest, InferResponse
from kserve.utils.utils import get_predict_response
from kserve_storage import Storage

PREDICTOR_METADATA_FILENAME = "predictor_metadata.json"

# Inference ``target`` column name always comes from ``TimeSeriesPredictor.target`` (see
# ``_target_column_from_predictor``). Optional non-empty env overrides apply only to id / time
# columns (``AUTOGLUON_TS_*``); see ``_load_ts_metadata``.
ENV_TS_ID_COLUMN = "AUTOGLUON_TS_ID_COLUMN"
ENV_TS_TIMESTAMP_COLUMN = "AUTOGLUON_TS_TIMESTAMP_COLUMN"


def _optional_env_nonempty(name: str) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


@dataclass
class TimeSeriesInferenceMetadata:
    target: str
    id_column: str
    timestamp_column: str
    prediction_length: int
    known_covariates_names: List[str]


def _nonempty_metadata_str(value: Any, *, field: str, meta_path: str) -> str:
    if value is None:
        raise InferenceError(
            f"{PREDICTOR_METADATA_FILENAME!r} ({meta_path}) is missing required string field {field!r}."
        )
    s = str(value).strip()
    if not s:
        raise InferenceError(
            f"{PREDICTOR_METADATA_FILENAME!r} ({meta_path}) has empty string field {field!r}."
        )
    return s


def _known_covariates_from_predictor(predictor: TimeSeriesPredictor) -> List[str]:
    known_raw = getattr(predictor, "known_covariates_names", None) or []
    if isinstance(known_raw, (list, tuple)):
        return [str(x) for x in known_raw]
    return []


def _target_column_from_predictor(
    predictor: TimeSeriesPredictor, *, meta_path: Optional[str] = None
) -> str:
    """Resolve the history/target column name strictly from the loaded ``TimeSeriesPredictor``."""
    raw_target = getattr(predictor, "target", None)
    if raw_target is None:
        suffix = f" ({meta_path})" if meta_path else ""
        raise InferenceError(
            "TimeSeriesPredictor.target is not set; cannot resolve the inference target column name"
            f"{suffix}."
        )
    s = str(raw_target).strip()
    if not s:
        suffix = f" ({meta_path})" if meta_path else ""
        raise InferenceError(
            "TimeSeriesPredictor.target is empty; cannot resolve the inference target column name"
            f"{suffix}."
        )
    return s


def _read_predictor_metadata_json(meta_path: str) -> Dict[str, Any]:
    """Read ``predictor_metadata.json`` and return the top-level JSON object."""
    try:
        with open(meta_path, encoding="utf-8") as fp:
            raw = json.load(fp)
    except OSError as e:
        raise InferenceError(f"Cannot read {meta_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise InferenceError(f"Invalid JSON in {meta_path}: {e}") from e

    if not isinstance(raw, dict):
        raise InferenceError(
            f"{meta_path} must contain a JSON object at the top level, got {type(raw).__name__}."
        )
    return raw


def _id_timestamp_columns_from_metadata_dict(
    raw: Dict[str, Any], meta_path: str
) -> Tuple[str, str]:
    id_column = _nonempty_metadata_str(
        raw.get("id_column"), field="id_column", meta_path=meta_path
    )
    timestamp_column = _nonempty_metadata_str(
        raw.get("timestamp_column"), field="timestamp_column", meta_path=meta_path
    )
    return id_column, timestamp_column


def _apply_id_timestamp_env_overrides(
    id_column: str, timestamp_column: str
) -> Tuple[str, str]:
    """Non-empty ``AUTOGLUON_TS_ID_COLUMN`` / ``AUTOGLUON_TS_TIMESTAMP_COLUMN`` override JSON names."""
    if (env_id := _optional_env_nonempty(ENV_TS_ID_COLUMN)) is not None:
        id_column = env_id
    if (env_ts := _optional_env_nonempty(ENV_TS_TIMESTAMP_COLUMN)) is not None:
        timestamp_column = env_ts
    return id_column, timestamp_column


def _prediction_length_from_predictor(
    predictor: TimeSeriesPredictor, *, meta_path: Optional[str] = None
) -> int:
    """Horizon steps always follow the loaded ``TimeSeriesPredictor`` (not ``predictor_metadata.json``)."""
    pl = int(getattr(predictor, "prediction_length", 1) or 1)
    if pl < 1:
        prefix = f"{meta_path}: " if meta_path else ""
        raise InferenceError(f"{prefix}prediction_length must be >= 1, got {pl}.")
    return pl


def _raise_if_known_covariates_overlap_columns(
    known_list: List[str],
    target: str,
    id_column: str,
    timestamp_column: str,
    *,
    meta_path: Optional[str] = None,
) -> None:
    reserved = {id_column, timestamp_column, target}
    overlap = reserved.intersection(known_list)
    if not overlap:
        return
    detail = f"{sorted(overlap)}."
    if meta_path is not None:
        raise InferenceError(
            f"{meta_path}: known covariate names overlap id/timestamp/target columns: {detail}"
        )
    raise InferenceError(
        "known covariate names overlap id/timestamp/target columns: " + detail
    )


def _ts_metadata_without_file(
    predictor: TimeSeriesPredictor,
    model_dir: str,
    meta_path: str,
    known_list: List[str],
) -> TimeSeriesInferenceMetadata:
    """Build inference column metadata when ``predictor_metadata.json`` is absent."""
    logger.warning(
        "%r not found at %s (model_dir=%r). Using default inference column names.",
        PREDICTOR_METADATA_FILENAME,
        meta_path,
        model_dir,
    )
    target = _target_column_from_predictor(predictor)
    id_column = str(os.environ.get(ENV_TS_ID_COLUMN, "item_id"))
    timestamp_column = str(os.environ.get(ENV_TS_TIMESTAMP_COLUMN, "timestamp"))
    pl = _prediction_length_from_predictor(predictor)

    _raise_if_known_covariates_overlap_columns(
        known_list, target, id_column, timestamp_column
    )

    return TimeSeriesInferenceMetadata(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        prediction_length=pl,
        known_covariates_names=known_list,
    )


def _ts_metadata_from_json_file(
    predictor: TimeSeriesPredictor,
    meta_path: str,
    known_list: List[str],
) -> TimeSeriesInferenceMetadata:
    """Build inference column metadata from an on-disk ``predictor_metadata.json``."""
    raw = _read_predictor_metadata_json(meta_path)
    target = _target_column_from_predictor(predictor, meta_path=meta_path)
    id_column, timestamp_column = _id_timestamp_columns_from_metadata_dict(
        raw, meta_path
    )
    id_column, timestamp_column = _apply_id_timestamp_env_overrides(
        id_column, timestamp_column
    )
    pl = _prediction_length_from_predictor(predictor, meta_path=meta_path)
    _raise_if_known_covariates_overlap_columns(
        known_list, target, id_column, timestamp_column, meta_path=meta_path
    )
    return TimeSeriesInferenceMetadata(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        prediction_length=pl,
        known_covariates_names=known_list,
    )


def _load_ts_metadata(
    predictor: TimeSeriesPredictor, model_dir: str
) -> TimeSeriesInferenceMetadata:
    """
    Prefer ``predictor_metadata.json`` in the predictor save directory (next to ``predictor.pkl``).

    The inference ``target`` column name is always taken from ``TimeSeriesPredictor.target`` (never
    from the metadata file or environment).

    If that file is absent, ``id_column`` / ``timestamp_column`` default to ``item_id`` and
    ``timestamp``, optionally overridden by non-empty ``AUTOGLUON_TS_ID_COLUMN`` /
    ``AUTOGLUON_TS_TIMESTAMP_COLUMN``; ``prediction_length`` comes from the loaded predictor. A
    warning is logged that the metadata file was not found.

    When the JSON file exists, it supplies ``id_column`` and ``timestamp_column`` (required string
    fields); ``AUTOGLUON_TS_ID_COLUMN`` and ``AUTOGLUON_TS_TIMESTAMP_COLUMN`` may still override
    those if set to a non-empty string (after strip). ``prediction_length`` always comes from the
    loaded predictor (any value in the JSON file is ignored).

    Request payloads must use these exact column names in ``instances`` / ``known_covariates``.
    Known covariate *names* still come from the loaded predictor (not duplicated in the JSON).
    """
    meta_path = os.path.join(model_dir, PREDICTOR_METADATA_FILENAME)
    known_list = _known_covariates_from_predictor(predictor)

    if not os.path.isfile(meta_path):
        return _ts_metadata_without_file(predictor, model_dir, meta_path, known_list)

    return _ts_metadata_from_json_file(predictor, meta_path, known_list)


def _check_duplicate_columns(
    df: pd.DataFrame, context: str, *, detail: Optional[str] = None
) -> None:
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        msg = f"{context} has duplicate column names: {dup!r}."
        if detail:
            msg += " " + detail
        raise InferenceError(msg)


def _dataframe_to_tsdf(
    df: pd.DataFrame, meta: TimeSeriesInferenceMetadata
) -> TimeSeriesDataFrame:
    _check_duplicate_columns(
        df,
        "instances DataFrame",
        detail="Use unique keys in each row object.",
    )
    missing = {meta.id_column, meta.timestamp_column, meta.target} - set(df.columns)
    if missing:
        raise InferenceError(
            f"instances DataFrame is missing required columns {sorted(missing)}. "
            f"Expected id_column={meta.id_column!r}, timestamp_column={meta.timestamp_column!r}, "
            f"target={meta.target!r}."
        )
    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=meta.id_column,
        timestamp_column=meta.timestamp_column,
    )


def _known_covariates_to_tsdf(
    rows: List[Dict[str, Any]], meta: TimeSeriesInferenceMetadata
) -> TimeSeriesDataFrame:
    df = pd.DataFrame(rows)
    _check_duplicate_columns(df, "known_covariates DataFrame")
    required = {meta.id_column, meta.timestamp_column, *meta.known_covariates_names}
    missing = required - set(df.columns)
    if missing:
        raise InferenceError(
            f"known_covariates is missing columns {sorted(missing)}. "
            f"Required: id/timestamp and {meta.known_covariates_names}."
        )
    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column=meta.id_column,
        timestamp_column=meta.timestamp_column,
    )


def _payload_instances_to_dataframe(payload: Dict) -> pd.DataFrame:
    """Build history DataFrame from v1 JSON (list of row dicts)."""
    raw = payload.get("instances")
    if raw is None:
        raw = payload.get("inputs")
    if raw is None:
        raise InferenceError(
            "JSON body must include 'instances' (time series history rows)."
        )
    if len(raw) == 0:
        raise InferenceError("'instances' must be a non-empty array.")
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, list) and all(isinstance(r, dict) for r in raw):
        return pd.DataFrame(raw)
    return pd.DataFrame(raw)


def _forecast_to_records(forecasts: pd.DataFrame) -> List[Dict[str, Any]]:
    work = forecasts.reset_index().copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    records: List[Dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, float)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                out[k] = int(v)
            elif pd.isna(v):
                out[k] = None
            else:
                out[k] = v
        records.append(out)
    return records


class AutoGluonTimeSeriesModel(Model):
    """Serve AutoGluon ``TimeSeriesPredictor`` via KServe REST v1 JSON."""

    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.platform = "autogluon-timeseries"
        self.versions = ["1"]
        self.ready = False
        self._predictor: Optional[TimeSeriesPredictor] = None
        self._metadata: Optional[TimeSeriesInferenceMetadata] = None

    def load(self) -> bool:
        local = Storage.download(self.model_dir)
        if not os.path.isdir(local):
            raise ModelMissingError(local)
        self._predictor = TimeSeriesPredictor.load(local)
        self._metadata = _load_ts_metadata(self._predictor, local)
        self.ready = True
        return self.ready

    def get_input_types(self) -> List[Dict]:
        """Time series uses REST v1 JSON only in phase 1; no v2 tensor schema."""
        return []

    def get_output_types(self) -> List[Dict]:
        return []

    def predict(
        self,
        payload: Union[Dict, InferRequest],
        headers: Optional[Dict[str, str]] = None,
    ) -> Union[Dict, InferResponse]:
        if isinstance(payload, InferRequest):
            raise InferenceError(
                "AutoGluon Time Series supports REST v1 JSON only: POST "
                "/v1/models/{model_name}:predict with Content-Type application/json. "
                "Use 'instances' for history and optional 'known_covariates' for the horizon."
            )
        if self._predictor is None or self._metadata is None:
            raise InferenceError("model is not loaded")

        try:
            instances = _payload_instances_to_dataframe(payload)

            meta = self._metadata
            ts_data = _dataframe_to_tsdf(instances, meta)

            known_covariates = payload.get("known_covariates")
            kc_tsdf: Optional[TimeSeriesDataFrame] = None
            if meta.known_covariates_names:
                if not known_covariates:
                    raise InferenceError(
                        "This model was trained with known_covariates_names; "
                        "include a top-level 'known_covariates' array in the JSON body."
                    )
                kc_tsdf = _known_covariates_to_tsdf(known_covariates, meta)

            # use_cache=False: avoid writing prediction_cache under read-only model dirs (e.g. downloaded URI).
            forecasts = self._predictor.predict(
                ts_data, known_covariates=kc_tsdf, use_cache=False
            )
            records = _forecast_to_records(forecasts)
            return get_predict_response(payload, records, self.name)
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(str(e)) from e
