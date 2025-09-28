import os, json, yaml
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
from urllib.parse import urlparse, unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# =========================================================
# CONFIG (works both on Windows host and inside Docker)
# =========================================================

def _default_store_uri() -> str:
    """
    Prefer env if set. Else, if your Windows path exists use it.
    Else fall back to a relative local folder (works in Docker with volume mount).
    """
    env_uri = os.getenv("MLFLOW_STORE_URI")
    if env_uri:
        return env_uri

    # Your Windows absolute path (used on your laptop)
    win_path = Path(r"C:/Users/shani/VS Code/MLOPs/EmployeeAttrition/mlruns")
    if win_path.exists():
        return win_path.resolve().as_uri()

    # Generic local path (in container, mount host mlruns to /app/mlruns)
    return "file:./mlruns"

MLFLOW_STORE_URI = _default_store_uri()
MODEL_NAME  = os.getenv("MODEL_NAME", "employee-attrition-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yaml")

# Safe config load
TARGET = "Attrition"
CFG_SERVE_FEATURES: List[str] = []
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    TARGET = cfg.get("data", {}).get("target", TARGET)
    CFG_SERVE_FEATURES = cfg.get("data", {}).get("serve_features", []) or []
except Exception as e:
    print(f"[config] Could not read {CONFIG_PATH}: {e}")

# Apply URIs
mlflow.set_tracking_uri(MLFLOW_STORE_URI)
mlflow.set_registry_uri(MLFLOW_STORE_URI)

# =========================================================
# Helpers for feature-name discovery
# =========================================================
def _try_pipeline_feature_names(model) -> List[str]:
    feats: List[str] = []
    try:
        steps = getattr(model, "named_steps", {})
        pre = None
        for key in ("preprocessor", "pre", "preprocess", "preprocessing"):
            if key in steps:
                pre = steps[key]
                break
        if pre is None or not hasattr(pre, "transformers"):
            return []
        for _, _, cols in pre.transformers:
            if cols is None or cols == "drop":
                continue
            if isinstance(cols, (list, tuple)):
                feats.extend(list(cols))
        seen, ordered = set(), []
        for c in feats:
            if c not in seen:
                ordered.append(c); seen.add(c)
        return ordered
    except Exception:
        return []

def _local_path_from_uri(uri: str) -> Path:
    try:
        parsed = urlparse(uri)
        if parsed.scheme in ("file", ""):
            return Path(unquote(parsed.path)).resolve()
    except Exception:
        pass
    return Path()

def _try_signature_feature_names(model_version) -> List[str]:
    try:
        model_dir = _local_path_from_uri(model_version.source)
        mlmodel_path = model_dir / "MLmodel"
        if not mlmodel_path.exists():
            return []
        with open(mlmodel_path, "r") as f:
            meta = yaml.safe_load(f)
        sig = meta.get("signature")
        if not sig:
            return []
        sig_obj = json.loads(sig) if isinstance(sig, str) else sig
        inputs = sig_obj.get("inputs", [])
        cols = [i.get("name") for i in inputs if isinstance(i, dict) and i.get("name")]
        seen, ordered = set(), []
        for c in cols:
            if c not in seen:
                ordered.append(c); seen.add(c)
        return ordered
    except Exception:
        return []

def _split_tag_list(value: str) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    return [p for p in parts if p]

def resolve_expected_columns(sk_model, mv) -> Tuple[List[str], str]:
    cols = _try_pipeline_feature_names(sk_model)
    if cols:
        return cols, "pipeline"

    tag_cols = _split_tag_list(mv.tags.get("features_used", "") if mv else "")
    if tag_cols:
        return tag_cols, "tag:features_used"

    cols = _try_signature_feature_names(mv) if mv else []
    if cols:
        return cols, "signature"

    if CFG_SERVE_FEATURES:
        return list(CFG_SERVE_FEATURES), "config:serve_features"

    return [], "none"

# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="Employee Attrition Predictor", version="1.0")
templates = Jinja2Templates(directory="templates")

# Globals populated on startup
_loaded_model = None
_loaded_mv = None
_expected_columns: List[str] = []
_expected_source = "none"
_threshold = 0.5

@app.on_event("startup")
def load_model_on_startup():
    global _loaded_model, _loaded_mv, _expected_columns, _expected_source, _threshold

    client = MlflowClient()
    try:
        # Try to find the aliased version in the registry
        _loaded_mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        _threshold = float(_loaded_mv.tags.get("threshold", "0.5"))
        # Load the sklearn/pyfunc model via registry alias
        _loaded_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        _expected_columns, _expected_source = resolve_expected_columns(_loaded_model, _loaded_mv)
        if not _expected_columns:
            print("[startup] WARNING: Expected feature names could not be determined.")
        print(f"[startup] Model loaded. Version={_loaded_mv.version}, Threshold={_threshold}, Features={len(_expected_columns)} from {_expected_source}")
    except MlflowException as e:
        # Keep API up; /health will show model_loaded = False
        print(f"[startup] Model not available yet: {e}")
        _loaded_model = None
        _loaded_mv = None
        _expected_columns, _expected_source = resolve_expected_columns(None, None)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "expected_columns": _expected_columns,
            "threshold": _threshold,
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
        },
    )

class BatchRequest(BaseModel):
    records: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_uri": mlflow.get_tracking_uri(),
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_loaded": _loaded_model is not None,
        "version": getattr(_loaded_mv, "version", None),
        "source_uri": getattr(_loaded_mv, "source", None),
        "threshold": _threshold,
        "tags": getattr(_loaded_mv, "tags", {}),
        "expected_feature_count": len(_expected_columns),
        "expected_source": _expected_source,
    }

@app.get("/schema")
def schema():
    note = (
        f"Send RAW features listed here; '{TARGET}' is the target and must NOT be included. "
        "Missing fields can be null; your pipeline's imputers (if any) will handle them."
    )
    return {"expected_columns": _expected_columns, "source": _expected_source, "note": note}

@app.post("/predict")
def predict(batch: BatchRequest):
    if _loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health and ensure MLflow registry is mounted/configured.")

    if not _expected_columns:
        raise HTTPException(
            status_code=500,
            detail=("Could not determine expected feature names. "
                    "Expose them via pipeline/signature/model tag, or set data.serve_features in config.yaml.")
        )

    try:
        rows = [{col: rec.get(col, None) for col in _expected_columns} for rec in batch.records]
        df = pd.DataFrame(rows, columns=_expected_columns)

        # Works for sklearn pipeline or estimator with predict_proba
        proba = _loaded_model.predict_proba(df)[:, 1]
        preds = (proba >= _threshold).astype(int)

        return {
            "results": [
                {"probability_yes": float(p), "prediction": int(y)}
                for p, y in zip(proba, preds)
            ]
        }
    except Exception as e:
        print(f"[predict] Error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed. Error: {e}")
