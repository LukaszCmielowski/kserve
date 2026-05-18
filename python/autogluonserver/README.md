# AutoGluon Server

[AutoGluon](https://auto.gluon.ai/) server serves **TabularPredictor** and **TimeSeriesPredictor** models in KServe from a shared image (`kserve/autogluonserver`).

- **Tabular**: KServe inference protocol **v1 and v2**; for classification models you can optionally return **class probabilities** instead of predicted labels (see [Classification probabilities](#classification-probabilities-predict_proba) below).
- **Time series**: **REST v1 JSON only** (`POST /v1/models/{name}:predict`). v2 tensor payloads are not supported for time series in this release.

**Auto-detection.** At startup the server downloads `storageUri` and decides whether the artifact is tabular or time series. You do not configure the predictor type in YAML. It calls `TimeSeriesPredictor.load` on that directory; if loading fails, it calls `TabularPredictor.load` on the same path.

**`storageUri`.** Set this to the directory AutoGluon wrote when you saved the model—the same folder you would pass to `TabularPredictor.load(...)` or `TimeSeriesPredictor.load(...)`. For example, if training ended with `predictor.save("models/iris/")`, use `storageUri: "gs://my-bucket/models/iris/"`. Do not use the parent bucket, raw training data, or a file inside the save tree.

## Tabular models

Models must be saved with `TabularPredictor.save(path)` (a directory). The server loads that directory and converts request instances (list of dicts or list of lists) to a pandas `DataFrame` for `predict()` or `predict_proba()`.

`storageUri` must point at the directory produced by `TabularPredictor.save`.

### Classification probabilities (`predict_proba`)

By default the server calls AutoGluon’s `TabularPredictor.predict()` and returns the **predicted label** for each row (for example `yes` or `no`).

For **binary and multiclass** models you can instead call `TabularPredictor.predict_proba()`, which returns the model’s estimated **probability for each class** (values between 0 and 1; per row they sum to 1). This is useful when you need confidence scores, thresholds, or ranking rather than a single hard label.

Enable it by setting the environment variable `PREDICT_PROBA=true` on the predictor container (see [Environment](#environment)). The predictor must support `predict_proba` (typical for classification).

**v1** responses use one object per instance, with a key per class name and the probability as the value, for example:

```json
{
  "predictions": [
    { "yes": 0.61, "no": 0.39 },
    { "yes": 0.42, "no": 0.58 }
  ]
}
```

**v2** responses expose one `FP64` output tensor per class (names like `proba_yes`, `proba_no`); see `GET /v2/models/{name}` for the exact output names for your model.

## Time series models

Models must be saved with `TimeSeriesPredictor.save()` (a directory). Point `storageUri` at that **predictor directory** (the same path you would pass to `TimeSeriesPredictor.load`).

Column names for request JSON are taken from the loaded `TimeSeriesPredictor` where available. You can override id, timestamp, and target column names with environment variables (see below) if they are not sufficient.

### Time series JSON request (`:predict`)

**History** — top-level `instances`: array of JSON objects, one object per time step (long format), each including `target` and any covariates present in training history.

**Known covariates on the horizon** (only if the model was trained with known covariates): top-level `known_covariates`, same column names as training for those features, plus the configured id and timestamp columns, covering the forecast horizon steps per series.

Example (names must match your schema and env overrides):

```json
{
  "instances": [
    { "item_id": "A", "timestamp": "2024-01-01T00:00:00", "target": 12.3 },
    { "item_id": "A", "timestamp": "2024-01-02T00:00:00", "target": 11.1 }
  ],
  "known_covariates": [
    { "item_id": "A", "timestamp": "2024-01-03T00:00:00", "promo": 1 }
  ]
}
```

**Response**: `{"predictions": [ ... ]}` — list of objects with the same **id** and **timestamp** column names as the request (from `predictor_metadata.json`, env overrides, or defaults), plus `mean` and quantile columns (e.g. `"0.1"`) from the trained predictor.

Use `modelFormat.name: autogluon` in `InferenceService` for both tabular and time series; the **same** runtime image auto-detects the artifact type from the save directory (see above). `ClusterServingRuntime` advertises a single format, `autogluon`.

## Run AutoGluon Server Locally

Install the [kserve](../kserve) package first. To install this package’s dependencies for local development, run the following from this directory (same pattern as [sklearnserver](../sklearnserver/README.md)):

```bash
make dev_install
```

**Note:** The dependency `autogluon.tabular[all]` pulls in CatBoost, which in the current lock file only has wheels for **Python 3.10** on some platforms. If you see an error like *"Distribution catboost can't be installed because it doesn't have a source distribution or wheel for the current platform"*, use Python 3.10 for this project (e.g. `uv venv .venv --python 3.10`, activate it, then `make dev_install`). To install into an already-active virtualenv elsewhere (e.g. the repo root), use `uv sync --active --group test`.

Check that the server is available:

```bash
python -m autogluonserver
usage: __main__.py [-h] [--http_port HTTP_PORT] [--grpc_port GRPC_PORT]
                   --model_dir MODEL_DIR [--model_name MODEL_NAME]
__main__.py: error: the following arguments are required: --model_dir
```

The model can be on the local filesystem, or in S3-compatible object storage, Azure Blob Storage, or Google Cloud Storage.

## Deploy on KServe

### Tabular

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: autogluon-iris
spec:
  predictor:
    model:
      modelFormat:
        name: autogluon
      storageUri: "gs://your-bucket/autogluon-tabular-model/"
```

### Time series

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: autogluon-ts-forecast
spec:
  predictor:
    model:
      modelFormat:
        name: autogluon
      storageUri: "gs://your-bucket/path/to/timeseries-predictor-save/"
```

## Environment

- **`PREDICT_PROBA`** (tabular): set to `"true"` to return [class probabilities](#classification-probabilities-predict_proba) via `predict_proba()` instead of predicted labels via `predict()`.
- **`AUTOGLUON_TS_ID_COLUMN`**, **`AUTOGLUON_TS_TIMESTAMP_COLUMN`**, **`AUTOGLUON_TS_TARGET`**: override series id, timestamp, and target column names for time series JSON (defaults: `item_id`, `timestamp`, and predictor `target` or `target`).

## Development

Install development dependencies from this directory:

```bash
make dev_install
```

Run tests from this directory (discovery is limited to `tests/` via `pyproject.toml`):

```bash
make test
```

Run static type checks:

```bash
make type_check
```

An empty result from mypy indicates success.

## Building the AutoGluon Server Docker Image

From the **repository root**, use the same Makefile targets as the other predictor images (`KO_DOCKER_REPO` and `AUTOGLUON_IMG` come from `kserve-images.env`; override `TAG` as needed):

```shell
make docker-build-autogluon
make docker-push-autogluon
```

To use a different AutoGluon version, change the version in `autogluonserver/pyproject.toml` (e.g. `autogluon.tabular==1.5.0` and `autogluon.timeseries==1.5.0`) and rebuild with a versioned tag.

Equivalent manual build from the `python` directory (replace the image name with your registry and tag):

```shell
docker build -t your-registry/autogluonserver:latest -f autogluon.Dockerfile .
docker push your-registry/autogluonserver:latest
```

Update the InferenceService or KServe API configuration to use your image if needed.