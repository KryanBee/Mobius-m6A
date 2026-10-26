"""Portable fitting-population contracts; no research directories are required."""
import hashlib
import json
import re

import numpy as np

from sonic.common import KEYS
from sonic.data import validate_keys
from sonic.rich_features import _key_digest


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fitting_identity(sites):
    """Fingerprint supplied labelled sites without requiring gene annotation."""
    validate_keys(sites, "final fitting")
    if not {"label", "label_original"} <= set(sites):
        raise ValueError("Final fitting requires original and binary labels")
    original = sites.label_original.to_numpy(dtype=float)
    if (not np.isfinite(original).all() or ((original < 0) | (original > 1)).any()
            or not sites.label.isin([0, 1]).all()
            or not np.array_equal(sites.label.to_numpy(), (original > 0).astype(int))
            or set(sites.label) != {0, 1}):
        raise ValueError("Targets must be finite [0,1], with binary label = original > 0 and both classes")
    sources = sites.source_id.drop_duplicates().tolist()
    ordered = sites.sort_values(KEYS)
    return {"fitting_sites": len(sites), "fitting_keys_sha256": _key_digest(sites),
            "fitting_labels_sha256": object_digest(ordered[KEYS + ["label"]].to_dict("records")),
            "original_labels_sha256": object_digest(ordered[KEYS + ["label_original"]].to_dict("records")),
            "source_counts": {s: int(sites.source_id.eq(s).sum()) for s in sources},
            "positive_sites": int(sites.label.sum()), "sources": sources}


def validate_refit_context(context):
    """Validate the original frozen mixed context or a portable new fitting context."""
    required = {"schema_version", "population", "recipe_sha256", "fitting_sites",
                "fitting_keys_sha256", "fitting_labels_sha256", "original_labels_sha256",
                "source_counts", "positive_sites", "sources", "source_label_sha256",
                "feature_content_sha256", "label_rule", "diagnostic_subset"}
    if not isinstance(context, dict) or set(context) != required:
        raise ValueError("Invalid final refit population contract")
    sources = context["sources"]
    schema = context["schema_version"]
    if (schema not in (2, 3) or type(schema) is not int
            or not isinstance(sources, list) or not sources
            or any(not isinstance(s, str) or not s or s != s.strip() for s in sources)
            or len(set(sources)) != len(sources)
            or (schema == 2 and (context["population"] != "all_labelled_human_synthetic"
                                or sources != ["data0", "data1", "data2"]))
            or (schema == 3 and context["population"] != "all_supplied_labelled")
            or context["label_rule"] != "label_original > 0" or context["diagnostic_subset"] is not False
            or type(context["fitting_sites"]) is not int or context["fitting_sites"] <= 0
            or type(context["positive_sites"]) is not int
            or not 0 < context["positive_sites"] < context["fitting_sites"]
            or not isinstance(context["source_counts"], dict) or set(context["source_counts"]) != set(sources)
            or any(type(v) is not int or v <= 0 for v in context["source_counts"].values())
            or sum(context["source_counts"].values()) != context["fitting_sites"]):
        raise ValueError("Invalid final refit population contract")
    labels = context["source_label_sha256"]
    digests = [context[k] for k in required if k.endswith("sha256") and k != "source_label_sha256"]
    if (not isinstance(labels, list) or len(labels) != len(sources)
            or any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d) for d in digests + labels)
            or labels != sorted(labels)):
        raise ValueError("Invalid final refit population fingerprint")
    return context


def validate_refit_bundle(model_dir, bundle):
    if bundle.get("fit_mode") != "final_refit_fixed_steps":
        raise ValueError("Submission model requires final_refit_fixed_steps mode")
    context = validate_refit_context(bundle.get("refit_context", {}))
    steps = bundle.get("fixed_steps")
    if (bundle.get("model") not in ("bag_network", "embedding_xgboost")
            or type(steps) is not int or steps <= 0
            or bundle.get("preprocessing", {}).get("fixed_steps") != steps):
        raise ValueError("Invalid fixed-length final model contract")
    if bundle["model"] == "bag_network":
        from sonic.read_normalization import load_reference
        reference = load_reference(model_dir, bundle)
    else:
        from sonic.feature_pipeline import load_transform
        _, reference = load_transform(model_dir, bundle)
    if reference is None or any(reference[k] != context[k] for k in ("fitting_sites", "fitting_keys_sha256")):
        raise ValueError("Final fitted reference differs from declared fitting population")
