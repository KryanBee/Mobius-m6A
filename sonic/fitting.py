"""Portable supervised refitting of the selected seven-base hybrid recipe.

All supplied sites fit both the read encoder and the trees. No gene split,
validation-driven selection, research artifact, or pretrained weight is used.
"""

from copy import deepcopy
from datetime import datetime, timezone
import csv
import gzip
from pathlib import Path
import resource
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sonic.common import KEYS, ROOT, file_info, new_directory, provenance, read_json, sha256, utc_now, write_json
from sonic.data import FeatureData, artifact_identity, build_artifact, exact_join, validate_keys
from sonic.metrics import class_weights
from sonic.refitting import fitting_identity, object_digest, validate_refit_context


RECIPE_PATH = ROOT / "config" / "final_recipe.json"
ENCODER_EPOCHS = 18
TREE_COUNT = 269


def read_labels(path, source_id):
    """Read external CSV/TSV or course data.info targets without losing fractions.

    Source identity comes from the explicit dataset pair. An optional source_id
    column must agree; gene_id and other metadata do not become predictors.
    """
    path = Path(path)
    compressed = path.suffix.lower() == ".gz"
    suffix = path.with_suffix("").suffix.lower() if compressed else path.suffix.lower()
    # Original course files include data.info.labelled; other suffixes use CSV.
    separator = "\t" if suffix == ".tsv" else ","
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=separator), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Labels have an empty or duplicate header: {path}")
    required = {"transcript_id", "transcript_position", "label"}
    if not required <= set(header):
        raise ValueError(f"Labels require transcript_id, transcript_position, label: {path}")
    frame = pd.read_csv(
        path, sep=separator, keep_default_na=False, encoding="utf-8-sig",
        dtype={"source_id": str, "transcript_id": str},
    )
    if "source_id" in frame and not frame.source_id.eq(source_id).all():
        raise ValueError(f"Label source_id differs from dataset source {source_id}: {path}")
    frame["source_id"] = source_id
    validate_keys(frame, f"labels for {source_id}")
    try:
        values = pd.to_numeric(frame.label, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Labels must be numeric and finite in [0,1]: {path}") from error
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Labels must be finite in [0,1]: {path}")
    result = frame[KEYS].copy()
    result["label_original"] = values
    result["label"] = (values > 0).astype(int)
    return result


def join_labels(data, datasets):
    """Require exact site matches while preserving JSON/source/read order."""
    labels = pd.concat([read_labels(labels, source) for source, _, labels in datasets], ignore_index=True)
    fitting = exact_join(data.sites, labels, "training labels").sort_values("_row").reset_index(drop=True)
    fitting_identity(fitting)
    return fitting


def load_recipe():
    recipe = read_json(RECIPE_PATH)
    if (set(recipe) != {"schema_version", "kind", "label_rule", "minimum_reads",
                       "sequence_encoding", "encoder", "component"}
            or recipe.get("schema_version") != 1
            or recipe.get("kind") != "portable_fixed_hybrid_recipe"
            or recipe.get("label_rule") != "label_original > 0"
            or recipe.get("minimum_reads") != 20
            or recipe.get("sequence_encoding") != "seven-base positional one-hot (28 columns)"
            or any(set(recipe[c]) != {"fixed_steps", "config"} for c in ("encoder", "component"))
            or recipe["encoder"]["fixed_steps"] != ENCODER_EPOCHS
            or recipe["component"]["fixed_steps"] != TREE_COUNT):
        raise ValueError("Invalid selected final recipe")
    neural, tree = (recipe[c]["config"] for c in ("encoder", "component"))
    shared = {"seed": 1101, "threads": 1, "class_weight": "none"}
    expected_neural = {
        **shared, "model": "bag_network", "read_normalization": "motif_z_v1",
        "params": {"hidden_dims": [32, 16], "site_hidden_dims": [16], "dropout": 0.1,
                   "learning_rate": 0.001, "weight_decay": 0.0001,
                   "batch_sites": 32, "batch_reads": 20000},
    }
    expected_tree = {
        **shared, "model": "embedding_xgboost", "feature_recipe": "readstats_context_v1",
        "exclude_feature_groups": ["log_dwell_summaries"],
        "params": {"max_depth": 4, "min_child_weight": 5.0, "learning_rate": 0.05,
                   "n_estimators": TREE_COUNT, "subsample": 0.8, "colsample_bytree": 0.8,
                   "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0},
    }
    if neural != expected_neural or tree != expected_tree:
        raise ValueError("Selected hybrid recipe settings changed")
    return recipe


def _prepare_rich(data, sites, config):
    from sonic.rich_features import PreparedRichData, compute_local, fit_reference, recipe_schema, transform_sites

    schema = recipe_schema(config["exclude_feature_groups"])
    local = compute_local(data, sites, progress=True)
    reference = fit_reference(data, sites, local)
    frame = transform_sites(data, sites, schema, reference, local, progress=True)
    return PreparedRichData(frame, schema, reference), frame


def fit_components(data, fitting, recipe, context, run, run_id, code):
    """Fresh encoder, fresh references, then fixed trees; verify raw replay."""
    from sonic.embedding import EmbeddingData, extract, pooling_contract
    from sonic.prediction import load_bundle, predict_data
    from sonic.training import fit_model

    neural, tree = (deepcopy(recipe[c]["config"]) for c in ("encoder", "component"))
    neural_weights = class_weights(fitting.label, neural["class_weight"])
    tree_weights = class_weights(fitting.label, tree["class_weight"])
    encoder_run = run / "encoder_fit"
    encoder_run.mkdir()
    fit_model(data, fitting, None, neural, neural_weights, encoder_run, run_id + "-encoder", code,
              fixed_steps=recipe["encoder"]["fixed_steps"], refit_context=context)
    encoder_dir, encoder_bundle = load_bundle(encoder_run / "model")
    encoder_hash = sha256(encoder_dir / "bundle.json")
    rich, rows = _prepare_rich(data, fitting, tree)
    vectors = extract(encoder_dir, data, fitting)
    rows = rows.merge(vectors, on=KEYS, how="left", sort=False, validate="one_to_one")
    spec = {
        "kind": "final_refit_encoder", "path": "encoder", **pooling_contract("bag_network"),
        "run_id": encoder_bundle["run_id"], "bundle_sha256": encoder_hash, "refit_context": context,
    }
    prepared = EmbeddingData(rich, rows, encoder_dir, spec)
    fit_model(prepared, rows, None, tree, tree_weights, run, run_id, code,
              fixed_steps=recipe["component"]["fixed_steps"], refit_context=context)
    model, bundle = load_bundle(run / "model")
    if sha256(encoder_dir / "bundle.json") != encoder_hash:
        raise ValueError("Encoder changed while fitting trees")

    # Bounded parity checks use no accuracy metric and do not claim holdout testing.
    probe = fitting.groupby(["source_id", "label"], sort=False).head(2).copy()
    expected, _ = predict_data(model, prepared, probe)
    actual, _ = predict_data(model, data, probe.drop(columns=["label", "label_original", "gene_id"], errors="ignore"))
    if (not expected[KEYS].equals(actual[KEYS])
            or not np.array_equal(expected.score.to_numpy(), actual.score.to_numpy())):
        raise ValueError("Raw-input and prepared-feature predictions differ")
    write_json(run / "replay_checks.json", {
        "sites": len(probe), "exact_scores": True,
        "labels_or_gene_ids_required_for_prediction": False,
        "metrics_computed": False, "model_bundle_sha256": sha256(model / "bundle.json"),
    })
    shutil.copy2(encoder_run / "history.csv", run / "encoder_history.csv")
    shutil.rmtree(encoder_run)
    return bundle


def _datasets(datasets):
    if not datasets:
        raise ValueError("Supply at least one source, JSON and label file")
    normalized, sources = [], set()
    for item in datasets:
        if len(item) != 3:
            raise ValueError("Each dataset needs SOURCE_ID SIGNAL_JSON LABELS")
        source, signals, labels = item
        if not isinstance(source, str) or not source or source != source.strip():
            raise ValueError("Source IDs must be nonempty strings without surrounding whitespace")
        if source in sources:
            raise ValueError(f"Each dataset needs a unique source ID: {source}")
        sources.add(source)
        paths = [Path(p).resolve(strict=True) for p in (signals, labels)]
        if not all(p.is_file() for p in paths):
            raise ValueError(f"Signal and label inputs must be files: {source}")
        normalized.append((source, *paths))
    return normalized


def train_model(datasets, output, *, temp_dir=None):
    """Fit all supplied sites and atomically publish a new model directory."""
    datasets = _datasets(datasets)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if temp_dir is not None and not Path(temp_dir).is_dir():
        raise ValueError("--temp-dir must be an existing directory")
    recipe = load_recipe()
    recipe_hash = sha256(RECIPE_PATH)
    # Validate all label files before allocating read caches or starting fitting.
    for source, _, labels in datasets:
        read_labels(labels, source)
    input_files = [(source, file_info(signals), file_info(labels)) for source, signals, labels in datasets]
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-hybrid-" + uuid.uuid4().hex[:8]
    code = provenance()
    with new_directory(output) as run, tempfile.TemporaryDirectory(prefix="sonic-training-", dir=temp_dir) as scratch:
        folders = []
        for index, (source, signals, _) in enumerate(datasets):
            folder = Path(scratch) / f"source-{index}"
            print(f"Reading source {source}", flush=True)
            build_artifact(folder, "bags", json_path=signals, source_id=source)
            folders.append(folder)
        data = FeatureData(folders)
        fitting = join_labels(data, datasets)
        if fitting.read_count.lt(recipe["minimum_reads"]).any():
            raise ValueError("Training requires at least 20 reads at every site")
        context = {
            "schema_version": 3, "population": "all_supplied_labelled",
            "recipe_sha256": object_digest(recipe), **fitting_identity(fitting),
            "diagnostic_subset": False, "label_rule": "label_original > 0",
            "source_label_sha256": sorted(labels["sha256"] for _, _, labels in input_files),
            "feature_content_sha256": object_digest(sorted(artifact_identity(m) for m in data.manifests)),
        }
        validate_refit_context(context)
        print(f"Fitting {len(fitting)} sites across {len(datasets)} sources: "
              f"{recipe['encoder']['fixed_steps']} encoder epochs, "
              f"{recipe['component']['fixed_steps']} trees, one CPU thread", flush=True)
        write_json(run / "recipe.json", recipe)
        with threadpool_limits(limits=1):
            fit_components(data, fitting, recipe, context, run, run_id, code)
        for (_, signals, labels), (_, before_signals, before_labels) in zip(datasets, input_files):
            if file_info(signals) != before_signals or file_info(labels) != before_labels:
                raise ValueError("A training input changed during fitting")
        if sha256(RECIPE_PATH) != recipe_hash:
            raise ValueError("The fixed recipe changed during fitting")
        write_json(run / "manifest.json", {
            "schema_version": 1, "kind": "portable_fixed_hybrid_fit", "status": "complete",
            "run_id": run_id, "finished_utc": utc_now(), "refit_context": context,
            "fitting_reads": int(fitting.read_count.sum()),
            "encoder_epochs": recipe["encoder"]["fixed_steps"],
            "tree_count": recipe["component"]["fixed_steps"],
            "input_files": [{"source_id": source,
                             "signals": {"name": Path(signals["path"]).name, "bytes": signals["bytes"], "sha256": signals["sha256"]},
                             "labels": {"name": Path(labels["path"]).name, "bytes": labels["bytes"], "sha256": labels["sha256"]}}
                            for source, signals, labels in input_files],
            "source_order": [source for source, _, _ in datasets],
            "row_order": "source argument order, then JSON site and read order",
            "gene_ids_required": False, "hyperparameters_retuned": False, "metrics_computed": False,
            "source_weighting": "none; unit site weights in both training stages", "provenance": code,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "outputs": {str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                        for p in sorted(run.rglob("*")) if p.is_file()},
        })
    return output
