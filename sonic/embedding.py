"""Frozen read representations plus the selected rich XGBoost inputs."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sonic.common import KEYS, sha256
from sonic.features import SEQUENCE_FEATURES
from sonic.rich_features import _sites

VERSION = "readstats_read_embedding_v1"
WIDTH = 16
COLUMNS = [f"learned_read_z{i}" for i in range(WIDTH)]
REPRESENTATIONS = {"bag_network": "mean_pooled_read_encoder_before_site_head"}

def pooling_contract(model, pooling="mean", width=WIDTH):
    if model != "bag_network" or pooling != "mean" or width != WIDTH:
        raise ValueError("Submission encoder requires standalone mean pooling of 16 values")
    return {"columns": COLUMNS.copy(), "representation": REPRESENTATIONS[model]}


def validate_pooling(spec, model, width=WIDTH):
    contract = pooling_contract(model, spec.get("pooling", "mean"), width)
    fields = {k: spec[k] for k in ("columns", "representation", "pooling", "sd_ddof", "sd_accumulation", "encoder_width") if k in spec}
    if fields != contract:
        raise ValueError("Frozen encoder pooling contract is invalid")
    return contract


def bundle_schema(spec):
    validate_pooling(spec, "bag_network")
    return 5


def validate_encoder(folder, bundle, *, allow_refit=False):
    from sonic.read_normalization import load_reference
    from sonic.refitting import validate_refit_context
    if (bundle.get("model") != "bag_network" or bundle.get("schema_version") != 3
            or bundle.get("read_normalization") != "motif_z_v1"
            or bundle["params"]["hidden_dims"][-1] != 16
            or bundle.get("fit_mode") != "final_refit_fixed_steps" or not allow_refit):
        raise ValueError("Expected the final motif-normalized read encoder with 16 pooled values")
    reference = load_reference(folder, bundle)
    context = validate_refit_context(bundle.get("refit_context", {}))
    if any(reference[k] != context[k] for k in ("fitting_sites", "fitting_keys_sha256")):
        raise ValueError("Final read encoder reference differs from its fitting population")
    return reference


def extract(folder, data, sites, *, pooling="mean"):
    from sonic.prediction import load_bundle, predict_data
    _sites(data, sites)
    folder, bundle = load_bundle(folder)
    validate_encoder(folder, bundle, allow_refit=True)
    columns = pooling_contract(bundle["model"], pooling, bundle["params"]["hidden_dims"][-1])["columns"]
    _, vectors = predict_data(folder, data, sites, representations=True)
    latent = [f"z{i}" for i in range(WIDTH)]
    if (not vectors[KEYS].equals(sites[KEYS].reset_index(drop=True))
            or [c for c in vectors if c.startswith("z")] != latent
            or not np.isfinite(vectors[latent].to_numpy()).all()):
        raise ValueError("Frozen encoder returned invalid or misaligned representations")
    return vectors[KEYS + latent].rename(columns=dict(zip(latent, columns)))


class EmbeddingData:
    kind, version = "summaries", VERSION

    def __init__(self, rich, sites, encoder_dir, spec):
        self.rich, self.sites = rich, sites
        self.schema, self.reference, self.transform_id = rich.schema, rich.reference, rich.transform_id
        self.numeric = rich.numeric + spec["columns"]
        self.encoder_dir, self.encoder_spec = Path(encoder_dir), spec


def save_encoder(model_dir, data):
    from sonic.prediction import load_bundle
    folder = data.encoder_dir
    if sha256(folder / "bundle.json") != data.encoder_spec["bundle_sha256"]:
        raise ValueError("Frozen encoder changed during feature extraction")
    _, bundle = load_bundle(folder)
    target = model_dir / "encoder"
    target.mkdir()
    for name in ["bundle.json", *bundle["files"]]:
        source, destination = folder / name, target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError("Frozen encoder copy changed")
    return dict(data.encoder_spec)


def validate_bundle(model_dir, bundle):
    from sonic.feature_pipeline import load_transform
    from sonic.prediction import load_bundle
    if bundle["schema_version"] != 5 or bundle["model"] != "embedding_xgboost":
        raise ValueError("Frozen embedding trees require bundle schema 5")
    schema, reference = load_transform(model_dir, bundle)
    spec = bundle.get("read_encoder", {})
    expected_schema = bundle_schema(spec)
    if bundle["schema_version"] != expected_schema:
        raise ValueError("Frozen pooling and bundle schema disagree")
    final = spec.get("kind") == "final_refit_encoder"
    if (spec.get("path") != "encoder" or not isinstance(spec.get("columns"), list)
            or (not final and (type(spec.get("fold")) is not int or spec["fold"] not in range(5)))
            or "encoder/bundle.json" not in bundle["files"]
            or bundle["files"]["encoder/bundle.json"] != spec.get("bundle_sha256")
            or bundle["numeric_features"] != schema["numeric_features"] + spec["columns"]
            or bundle.get("tabular_predictors") != schema["numeric_features"] + SEQUENCE_FEATURES
            or bundle["preprocessing"].get("scaler") is not None):
        raise ValueError("Invalid frozen encoder feature contract")
    folder, encoder = load_bundle(model_dir / "encoder")
    read_reference = validate_encoder(folder, encoder, allow_refit=final)
    validate_pooling(spec, encoder["model"], encoder["params"]["hidden_dims"][-1])
    if final:
        from sonic.refitting import validate_refit_context
        context = spec.get("refit_context", {})
        validate_refit_context(context)
        if (bundle.get("fit_mode") != "final_refit_fixed_steps" or "fold" in spec
                or encoder.get("fit_mode") != "final_refit_fixed_steps"
                or bundle.get("refit_context") != context or encoder.get("refit_context") != context):
            raise ValueError("Final hybrid and encoder fitting contracts disagree")
    elif bundle.get("fit_mode") == "final_refit_fixed_steps" or "refit_context" in bundle:
        raise ValueError("Research hybrid cannot declare final refit metadata")
    if (encoder["run_id"] != spec.get("run_id") or reference is None
            or any(reference[k] != read_reference[k] for k in
                   ("fitting_sites", "fitting_reads", "fitting_keys_sha256"))):
        raise ValueError("Frozen encoder and tree references use different fitting populations")


def prediction_frame(model_dir, bundle, data, sites):
    from sonic.feature_pipeline import prediction_frame as rich_frame
    if isinstance(data, EmbeddingData):
        if (data.encoder_spec != bundle["read_encoder"]
                or data.transform_id != bundle["feature_transform"]["transform_sha256"]):
            raise ValueError("Prepared embeddings use a different encoder or fitted reference")
        columns = KEYS + ["sequence_7mer"] + data.numeric
        result = sites[KEYS].merge(data.sites[columns], on=KEYS, how="left", sort=False,
                                  validate="one_to_one", indicator=True)
        if not result._merge.eq("both").all():
            raise ValueError("Prepared embeddings do not cover requested sites")
        return result.drop(columns="_merge")
    metadata = sites[KEYS + ["sequence_7mer", "read_count", "_row"]]
    _sites(data, metadata)
    frame = rich_frame(model_dir, bundle, data, metadata)
    vectors = extract(model_dir / "encoder", data, metadata, pooling=bundle["read_encoder"].get("pooling", "mean"))
    return frame.merge(vectors, on=KEYS, how="left", sort=False, validate="one_to_one")
