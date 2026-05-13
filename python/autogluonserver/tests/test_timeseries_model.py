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

import json
import logging

import pandas as pd
import pytest
from kserve.errors import InferenceError
from kserve.protocol.infer_type import InferRequest, InferInput

from autogluonserver.timeseries_model import AutoGluonTimeSeriesModel


def _write_predictor_metadata(
    tmp_path,
    *,
    target: str = "y",
    id_column: str = "item_id",
    timestamp_column: str = "ts",
    prediction_length: int = 1,
) -> None:
    payload = {
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "prediction_length": prediction_length,
    }
    (tmp_path / "predictor_metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


class FakeTimeSeriesPredictor:
    def __init__(self, known_covariates_names=None):
        self.last_data = None
        self.last_known_covariates = None
        self.target = "y"
        self.prediction_length = 1
        self.known_covariates_names = known_covariates_names or []

    def predict(self, data, known_covariates=None, **kwargs):
        self.last_data = data
        self.last_known_covariates = known_covariates
        idx = pd.MultiIndex.from_tuples(
            [("i1", pd.Timestamp("2024-01-05"))],
            names=["item_id", "timestamp"],
        )
        return pd.DataFrame({"mean": [3.14], "0.1": [2.0]}, index=idx)


def test_timeseries_load_and_predict_v1(monkeypatch, tmp_path):
    _write_predictor_metadata(tmp_path)
    fake = FakeTimeSeriesPredictor()
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: fake,
    )

    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    assert model.load()
    resp = model.predict(
        {
            "instances": [
                {"item_id": "i1", "ts": "2024-01-01", "y": 1.0},
                {"item_id": "i1", "ts": "2024-01-02", "y": 2.0},
            ]
        }
    )
    assert fake.last_known_covariates is None
    assert "predictions" in resp
    assert len(resp["predictions"]) == 1
    row = resp["predictions"][0]
    assert row["item_id"] == "i1"
    assert row["mean"] == pytest.approx(3.14)
    assert row["0.1"] == pytest.approx(2.0)


def test_timeseries_known_covariates_passed(monkeypatch, tmp_path):
    _write_predictor_metadata(tmp_path)
    fake = FakeTimeSeriesPredictor(known_covariates_names=["promo"])
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: fake,
    )
    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    model.load()
    model.predict(
        {
            "instances": [{"item_id": "i1", "ts": "2024-01-01", "y": 1.0}],
            "known_covariates": [
                {"item_id": "i1", "ts": "2024-01-05", "promo": 1},
            ],
        }
    )
    assert fake.last_known_covariates is not None


def test_timeseries_missing_known_covariates_raises(monkeypatch, tmp_path):
    _write_predictor_metadata(tmp_path)
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: FakeTimeSeriesPredictor(known_covariates_names=["promo"]),
    )
    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    model.load()
    with pytest.raises(InferenceError, match="known_covariates"):
        model.predict({"instances": [{"item_id": "i1", "ts": "2024-01-01", "y": 1.0}]})


def test_timeseries_without_metadata_json_uses_default_columns(
    caplog, monkeypatch, tmp_path
):
    fake = FakeTimeSeriesPredictor()
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: fake,
    )
    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    with caplog.at_level(logging.WARNING, logger="kserve"):
        assert model.load()
    assert any(
        "predictor_metadata.json" in r.message
        and "default inference column names" in r.message
        for r in caplog.records
    )
    resp = model.predict(
        {
            "instances": [
                {"item_id": "i1", "timestamp": "2024-01-01", "y": 1.0},
                {"item_id": "i1", "timestamp": "2024-01-02", "y": 2.0},
            ]
        }
    )
    assert fake.last_data is not None
    assert "predictions" in resp


def test_timeseries_env_overrides_column_names(monkeypatch, tmp_path):
    _write_predictor_metadata(tmp_path)
    monkeypatch.setenv("AUTOGLUON_TS_ID_COLUMN", "series_id")
    monkeypatch.setenv("AUTOGLUON_TS_TIMESTAMP_COLUMN", "time")
    fake = FakeTimeSeriesPredictor()
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: fake,
    )
    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    assert model.load()
    model.predict(
        {
            "instances": [
                {"series_id": "i1", "time": "2024-01-01", "y": 1.0},
                {"series_id": "i1", "time": "2024-01-02", "y": 2.0},
            ]
        }
    )
    assert fake.last_data is not None


def test_timeseries_v2_request_raises(monkeypatch, tmp_path):
    _write_predictor_metadata(tmp_path)
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.Storage.download", lambda _: str(tmp_path)
    )
    monkeypatch.setattr(
        "autogluonserver.timeseries_model.TimeSeriesPredictor.load",
        lambda path: FakeTimeSeriesPredictor(),
    )
    model = AutoGluonTimeSeriesModel("forecast", "s3://bucket/artifact")
    model.load()
    req = InferRequest(
        model_name="forecast",
        infer_inputs=[
            InferInput(name="x", shape=[1], datatype="FP64", data=[1.0]),
        ],
    )
    with pytest.raises(InferenceError, match="REST v1 JSON"):
        model.predict(req)
