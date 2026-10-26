"""Validated, bounded-memory prediction from the portable final hybrid model."""
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sonic.common import KEYS, ROOT, read_json, sha256
from sonic.features import SEQUENCE_ENCODING, SEQUENCE_FEATURES, READ_FEATURES
from sonic.metrics import validate_scores

OUTPUT_COLUMNS = ["transcript_id", "transcript_position", "score"]


def load_bundle(path):
    """Verify all checkpoint bytes, feature schemas and fitting references."""
    model_dir = Path(path).resolve(strict=True)
    bundle = read_json(model_dir / "bundle.json")
    supported = {("bag_network", 3), ("embedding_xgboost", 5)}
    if ((bundle.get("model"), bundle.get("schema_version")) not in supported
            or bundle.get("sequence_encoding") != SEQUENCE_ENCODING
            or not isinstance(bundle.get("numeric_features"), list)
            or bundle.get("predictors") != bundle["numeric_features"] + SEQUENCE_FEATURES
            or type(bundle.get("threads")) is not int or bundle["threads"] < 1):
        raise ValueError("Unsupported model bundle or predictor schema")
    required = ({"weights.pt", "read_motif_reference.json"} if bundle["model"] == "bag_network" else
                {"trees.ubj", "feature_recipe.json", "motif_reference.json", "encoder/bundle.json",
                 "encoder/weights.pt", "encoder/read_motif_reference.json"})
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != required:
        raise ValueError("Model checkpoint files do not match its schema")
    for name, digest in files.items():
        target = (model_dir / name).resolve()
        if (not target.is_relative_to(model_dir) or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(target) != digest):
            raise ValueError(f"Model checksum mismatch: {name}")
    from sonic.refitting import validate_refit_bundle
    validate_refit_bundle(model_dir, bundle)
    if bundle["model"] == "embedding_xgboost":
        from sonic.embedding import validate_bundle
        validate_bundle(model_dir, bundle)
    else:
        from sonic.embedding import validate_encoder
        validate_encoder(model_dir, bundle, allow_refit=True)
    return model_dir, bundle


def input_contract(bundle):
    return "bags", "reads9_v1"


def _check_data(data):
    if data.kind != "bags" or data.version != "reads9_v1" or data.numeric != READ_FEATURES:
        raise ValueError("Read features do not match the saved model input contract")


def predict_data(model_path, data, sites=None, representations=False):
    """Array interface used by training replay; does not read external labels."""
    model_dir, bundle = load_bundle(model_path)
    sites = data.sites if sites is None else sites
    if sites.empty:
        raise ValueError("No sites selected for prediction")
    vectors = None
    with threadpool_limits(limits=bundle["threads"]):
        if bundle["model"] == "bag_network":
            _check_data(data)
            from sonic import bag
            from sonic.read_normalization import MotifReadScaler, load_reference
            model = bag.load(model_dir, bundle)
            rows = sites["_row"].to_numpy()
            scaler = MotifReadScaler(data, rows, load_reference(model_dir, bundle))
            scores, vectors = bag.predict(model, data, rows, scaler, bundle["params"], representations)
        else:
            if representations:
                raise ValueError("Representation export is available for the encoder only")
            from sonic.embedding import EmbeddingData, prediction_frame
            if isinstance(data, EmbeddingData):
                from sonic import tabular
                frame = prediction_frame(model_dir, bundle, data, sites)
                scores = tabular.predict(tabular.load(model_dir, bundle), bundle, frame)
            else:
                predictor = HybridPredictor(model_dir)
                scores = predictor.scores(data, sites)
    predictions = sites[KEYS].reset_index(drop=True).copy()
    predictions["score"] = validate_scores(scores)
    encoded = None
    if vectors is not None:
        encoded = pd.concat([predictions[KEYS], pd.DataFrame(vectors, columns=[f"z{i}" for i in range(vectors.shape[1])])], axis=1)
        encoded["run_id"] = bundle["run_id"]
        encoded["checkpoint_sha256"] = bundle["files"]["weights.pt"]
    return predictions, encoded


class HybridPredictor:
    """Load the validated encoder and tree once, then reuse them for each chunk."""

    def __init__(self, model_path):
        from sonic import bag, tabular
        from sonic.feature_pipeline import load_transform
        from sonic.read_normalization import load_reference
        self.model_dir, self.bundle = load_bundle(model_path)
        if self.bundle["model"] != "embedding_xgboost":
            raise ValueError("Prediction requires the final hybrid model directory")
        self.schema, self.reference = load_transform(self.model_dir, self.bundle)
        encoder_dir, self.encoder_bundle = load_bundle(self.model_dir / "encoder")
        self.read_reference = load_reference(encoder_dir, self.encoder_bundle)
        self.encoder = bag.load(encoder_dir, self.encoder_bundle)
        self.tree = tabular.load(self.model_dir, self.bundle)

    def scores(self, data, sites=None):
        from sonic import bag, tabular
        from sonic.embedding import COLUMNS
        from sonic.read_normalization import MotifReadScaler
        from sonic.rich_features import transform_sites
        _check_data(data)
        sites = data.sites if sites is None else sites
        if sites.empty:
            raise ValueError("No sites selected for prediction")
        with threadpool_limits(limits=self.bundle["threads"]):
            frame = transform_sites(data, sites, self.schema, self.reference)
            rows = sites["_row"].to_numpy()
            scaler = MotifReadScaler(data, rows, self.read_reference)
            _, vectors = bag.predict(self.encoder, data, rows, scaler, self.encoder_bundle["params"], True)
            if vectors.shape != (len(sites), len(COLUMNS)) or not np.isfinite(vectors).all():
                raise ValueError("Frozen encoder returned invalid read representations")
            frame[COLUMNS] = vectors
            return validate_scores(tabular.predict(self.tree, self.bundle, frame))


def predict_file(input_path, output_path, model_path=None, *, chunk_sites=2048, chunk_reads=200000):
    """Publish one complete CSV atomically, refusing any existing output path.

    Only the current chunk of reads/features is retained. Duplicate-key detection
    retains one identifier pair per input site. Temporary output lives beside the
    destination so publication stays atomic, with no extra read-cache disk space.
    """
    from sonic.data import bag_chunks
    output = Path(output_path).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if any(type(value) is not int or value <= 0 for value in (chunk_sites, chunk_reads)):
        raise ValueError("Chunk sites and reads must be positive integers")
    predictor = HybridPredictor(ROOT / "model" if model_path is None else model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=".prediction-",
                                         suffix=".partial", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for data in bag_chunks(input_path, chunk_sites=chunk_sites, chunk_reads=chunk_reads):
                scores = predictor.scores(data)
                frame = data.sites[["transcript_id", "transcript_position"]].copy()
                frame["score"] = scores
                frame.to_csv(handle, index=False, header=count == 0)
                count += len(frame)
            if count == 0:
                raise ValueError("Input JSON contains no sites")
            handle.flush()
            os.fsync(handle.fileno())
        # link() fails if another process has created the destination meanwhile.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count
