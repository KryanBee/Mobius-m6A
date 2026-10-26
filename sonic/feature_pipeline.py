"""Portable saved contracts for the selected rich feature recipe."""
from sonic.common import KEYS, read_json, write_json
from sonic.features import READ_FEATURES
from sonic.rich_features import (VERSION, PreparedRichData, recipe_schema,
                                 transform_identity, transform_sites, validate_reference)

def save_transform(model_dir, data):
    write_json(model_dir / "feature_recipe.json", data.schema)
    write_json(model_dir / "motif_reference.json", data.reference)
    return {"recipe": VERSION, "schema_file": "feature_recipe.json", "reference_file": "motif_reference.json",
            "input_kind": "bags", "input_feature_version": "reads9_v1", "input_numeric_features": list(READ_FEATURES),
            "transform_sha256": data.transform_id}


def load_transform(model_dir, bundle):
    spec = bundle.get("feature_transform", {})
    if (spec.get("recipe") != VERSION or spec.get("input_kind") != "bags"
            or spec.get("input_feature_version") != "reads9_v1"
            or spec.get("input_numeric_features") != list(READ_FEATURES)
            or spec.get("schema_file") != "feature_recipe.json" or spec.get("reference_file") != "motif_reference.json"
            or not {"feature_recipe.json", "motif_reference.json"} <= bundle["files"].keys()):
        raise ValueError("Invalid saved rich-feature transform contract")
    schema = read_json(model_dir / spec["schema_file"])
    reference = read_json(model_dir / spec["reference_file"])
    numeric = bundle.get("tabular_numeric_features")
    from sonic.embedding import VERSION as EMBEDDING_VERSION
    version = EMBEDDING_VERSION
    if (bundle.get("model") != "embedding_xgboost"
            or schema != recipe_schema(schema["exclude_feature_groups"])
            or numeric != schema["numeric_features"]
            or bundle["feature_version"] != version):
        raise ValueError("Saved rich feature recipe/order disagrees with its model")
    if any(r["fitted_reference"] for r in schema["features"]):
        validate_reference(reference)
    elif reference is not None:
        raise ValueError("Unexpected motif reference for a non-contextual feature subset")
    if transform_identity(schema, reference) != spec["transform_sha256"]:
        raise ValueError("Saved feature transform identity mismatch")
    return schema, reference


def prediction_frame(model_dir, bundle, data, sites):
    schema, reference = load_transform(model_dir, bundle)
    if isinstance(data, PreparedRichData):
        if data.transform_id != bundle["feature_transform"]["transform_sha256"]:
            raise ValueError("Fold features were generated with a different fitted motif reference")
        # Do not trust user-supplied numeric columns for a selected set of keys.
        columns = KEYS + ["sequence_7mer"] + data.numeric
        selected = sites[KEYS].merge(data.sites[columns], on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
        if not selected._merge.eq("both").all():
            raise ValueError("Prediction sites missing from the prepared fold matrix")
        return selected.drop(columns="_merge")
    return transform_sites(data, sites, schema, reference)
