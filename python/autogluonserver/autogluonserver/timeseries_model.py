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
from typing import Any, Dict, List, Optional, Union

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

# When ``predictor_metadata.json`` exists: optional non-empty env overrides for target / id / time.
# When it is missing: default column names from predictor + ``AUTOGLUON_TS_*`` (see ``_load_ts_metadata``).
ENV_TS_TARGET = "AUTOGLUON_TS_TARGET"
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


def _load_ts_metadata(
    predictor: TimeSeriesPredictor, model_dir: str
) -> TimeSeriesInferenceMetadata:
    """
    Prefer ``predictor_metadata.json`` in the predictor save directory (next to ``predictor.pkl``).

    If that file is absent, use default column names: ``target`` from ``predictor.target``,
    then ``AUTOGLUON_TS_TARGET``, then ``"target"``; ``id_column`` / ``timestamp_column`` from
    ``AUTOGLUON_TS_ID_COLUMN`` / ``AUTOGLUON_TS_TIMESTAMP_COLUMN`` defaulting to ``item_id`` and
    ``timestamp``; ``prediction_length`` from the loaded predictor. A warning is logged that the
    metadata file was not found.

    When the JSON file exists, ``AUTOGLUON_TS_TARGET``, ``AUTOGLUON_TS_ID_COLUMN``, and
    ``AUTOGLUON_TS_TIMESTAMP_COLUMN`` may still override the corresponding JSON fields if set to a
    non-empty string (after strip).

    Request payloads must use these exact column names in ``instances`` / ``known_covariates``.
    Known covariate *names* still come from the loaded predictor (not duplicated in the JSON).
    """
    meta_path = os.path.join(model_dir, PREDICTOR_METADATA_FILENAME)
    known_list = _known_covariates_from_predictor(predictor)

    if not os.path.isfile(meta_path):
        logger.warning(
            "%r not found at %s (model_dir=%r). Using default inference column names.",
            PREDICTOR_METADATA_FILENAME,
            meta_path,
            model_dir,
        )
        target_raw = getattr(predictor, "target", None) or os.environ.get(
            ENV_TS_TARGET, "target"
        )
        target = str(target_raw)
        id_column = str(os.environ.get(ENV_TS_ID_COLUMN, "item_id"))
        timestamp_column = str(os.environ.get(ENV_TS_TIMESTAMP_COLUMN, "timestamp"))
        pl = int(getattr(predictor, "prediction_length", 1) or 1)
        if pl < 1:
            raise InferenceError(f"prediction_length must be >= 1, got {pl}.")

        reserved = {id_column, timestamp_column, target}
        overlap = reserved.intersection(known_list)
        if overlap:
            raise InferenceError(
                "known covariate names overlap id/timestamp/target columns: "
                f"{sorted(overlap)}."
            )

        return TimeSeriesInferenceMetadata(
            target=target,
            id_column=id_column,
            timestamp_column=timestamp_column,
            prediction_length=pl,
            known_covariates_names=known_list,
        )

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

    target = _nonempty_metadata_str(
        raw.get("target"), field="target", meta_path=meta_path
    )
    id_column = _nonempty_metadata_str(
        raw.get("id_column"), field="id_column", meta_path=meta_path
    )
    timestamp_column = _nonempty_metadata_str(
        raw.get("timestamp_column"), field="timestamp_column", meta_path=meta_path
    )

    if (env_target := _optional_env_nonempty(ENV_TS_TARGET)) is not None:
        target = env_target
    if (env_id := _optional_env_nonempty(ENV_TS_ID_COLUMN)) is not None:
        id_column = env_id
    if (env_ts := _optional_env_nonempty(ENV_TS_TIMESTAMP_COLUMN)) is not None:
        timestamp_column = env_ts

    if "prediction_length" in raw and raw["prediction_length"] is not None:
        try:
            pl = int(raw["prediction_length"])
        except (TypeError, ValueError) as e:
            raise InferenceError(
                f"{meta_path}: prediction_length must be an integer, got {raw['prediction_length']!r}."
            ) from e
    else:
        pl = int(getattr(predictor, "prediction_length", 1) or 1)

    if pl < 1:
        raise InferenceError(f"{meta_path}: prediction_length must be >= 1, got {pl}.")

    pred_target = getattr(predictor, "target", None)
    if pred_target is not None and str(pred_target) != target:
        raise InferenceError(
            f"{meta_path}: field target={target!r} does not match loaded predictor.target={str(pred_target)!r}."
        )

    pred_pl = getattr(predictor, "prediction_length", None)
    if (
        "prediction_length" in raw
        and raw["prediction_length"] is not None
        and pred_pl is not None
        and int(pred_pl) != pl
    ):
        raise InferenceError(
            f"{meta_path}: field prediction_length={pl} does not match loaded "
            f"predictor.prediction_length={int(pred_pl)}."
        )

    reserved = {id_column, timestamp_column, target}
    overlap = reserved.intersection(known_list)
    if overlap:
        raise InferenceError(
            f"{meta_path}: known covariate names overlap id/timestamp/target columns: {sorted(overlap)}."
        )

    return TimeSeriesInferenceMetadata(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        prediction_length=pl,
        known_covariates_names=known_list,
    )


def _dataframe_to_tsdf(
    df: pd.DataFrame, meta: TimeSeriesInferenceMetadata
) -> TimeSeriesDataFrame:
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        raise InferenceError(
            f"instances DataFrame has duplicate column names: {dup!r}. "
            "Use unique keys in each row object."
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
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        raise InferenceError(
            f"known_covariates DataFrame has duplicate column names: {dup!r}."
        )
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
