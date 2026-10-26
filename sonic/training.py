"""Fit and export the selected encoder or hybrid at a fixed training length."""

from sonic.common import sha256, write_json
from sonic.features import SEQUENCE_ENCODING, SEQUENCE_FEATURES


def fit_model(data, train, validation, config, weighting, run, run_id,
              code_provenance, *, fixed_steps, refit_context):
    """Fit fresh parameters on all supplied rows, without model selection."""
    from sonic.refitting import fitting_identity, validate_refit_context

    if type(fixed_steps) is not int or fixed_steps <= 0 or validation is not None:
        raise ValueError("Fixed-recipe fitting needs positive steps and no stopping data")
    validate_refit_context(refit_context)
    if (refit_context["schema_version"] != 3
            or any(refit_context[k] != v for k, v in fitting_identity(train).items())):
        raise ValueError("Refit context differs from the supplied fitting population")
    if config["model"] not in ("bag_network", "embedding_xgboost"):
        raise ValueError("Only the selected encoder and hybrid are supported")

    model_dir = run / "model"
    model_dir.mkdir()
    feature_transform, encoder = None, None
    if config["model"] == "embedding_xgboost":
        from sonic.embedding import EmbeddingData, save_encoder
        from sonic.feature_pipeline import save_transform
        from sonic.rich_features import recipe_schema

        if (not isinstance(data, EmbeddingData)
                or data.schema != recipe_schema(config["exclude_feature_groups"])):
            raise ValueError("Hybrid fitting requires the prepared feature matrix")
        if (data.encoder_spec.get("kind") != "final_refit_encoder"
                or data.encoder_spec.get("refit_context") != refit_context
                or data.encoder_spec.get("pooling", "mean") != "mean"):
            raise ValueError("Hybrid needs its own fresh mean-pooling encoder")
        feature_transform = save_transform(model_dir, data)
        encoder = save_encoder(model_dir, data)
        from sonic import tabular
        preprocessing, history = tabular.fit(
            data, train, None, config, weighting, model_dir, fixed_steps=fixed_steps)
    else:
        from sonic import bag
        preprocessing, history = bag.fit(
            data, train, None, config, weighting, model_dir, run / "history.csv",
            fixed_steps=fixed_steps)
    history.to_csv(run / "history.csv", index=False)
    bundle = {
        "schema_version": 3 if encoder is None else 5,
        "run_id": run_id,
        "model": config["model"],
        "feature_version": data.version,
        "numeric_features": data.numeric,
        "sequence_encoding": SEQUENCE_ENCODING,
        "predictors": data.numeric + SEQUENCE_FEATURES,
        "params": config["params"],
        "threads": config["threads"],
        "seed": config["seed"],
        "preprocessing": preprocessing,
        "weighting": weighting,
        "files": {str(p.relative_to(model_dir)): sha256(p)
                  for p in sorted(model_dir.rglob("*")) if p.is_file()},
        "provenance": code_provenance,
        "fit_mode": "final_refit_fixed_steps",
        "fixed_steps": fixed_steps,
        "refit_context": refit_context,
    }
    if encoder is None:
        bundle["read_normalization"] = "motif_z_v1"
    else:
        bundle.update(
            feature_transform=feature_transform,
            read_encoder=encoder,
            tabular_numeric_features=data.rich.numeric,
            tabular_predictors=data.rich.numeric + SEQUENCE_FEATURES,
        )
    write_json(model_dir / "bundle.json", bundle)
    return preprocessing, history
