"""XGBoost fitting, scoring and portable checkpoints for site-level summaries."""

import numpy as np
import pandas as pd
from sonic.features import encode_sequence
from sonic.preprocessing import transform
from sonic.config import TREE_MODELS


def design_matrix(sites, numeric, scaler):
    return np.column_stack((transform(sites[numeric].to_numpy(), scaler),
                            encode_sequence(sites["sequence_7mer"])))


def fit(data, train, validation, config, weighting, model_dir, *, fixed_steps=None):
    import xgboost as xgb

    if config["model"] not in TREE_MODELS:
        raise ValueError("Tabular fitting supports the selected hybrid only")
    if validation is not None or type(fixed_steps) is not int or fixed_steps <= 0:
        raise ValueError("Submission training requires a positive fixed tree count, without a stopping set")
    params = config["params"]
    x_train = design_matrix(train, data.numeric, None)
    labels = train["label"].to_numpy(dtype=int)
    weights = np.asarray(weighting["weights"])[labels]
    metadata = {"scaler": None, "scaling_unit": "none"}
    settings = {k: v for k, v in params.items() if k != "patience"}
    settings["n_estimators"] = fixed_steps
    estimator = xgb.XGBClassifier(
        **settings, objective="binary:logistic", tree_method="hist", device="cpu",
        n_jobs=config["threads"], random_state=config["seed"], disable_default_eval_metric=True)
    estimator.fit(x_train, labels, sample_weight=weights, verbose=False)
    estimator.save_model(model_dir / "trees.ubj")
    metadata.update(trees=int(estimator.get_booster().num_boosted_rounds()),
                    fixed_steps=fixed_steps, stop_reason="fixed_steps")
    return metadata, pd.DataFrame([{"trees": metadata["trees"]}])


def load(model_dir, bundle):
    if bundle["model"] not in TREE_MODELS:
        raise ValueError("Tabular checkpoints require xgboost")
    import xgboost as xgb
    estimator = xgb.XGBClassifier(n_jobs=bundle["threads"], device="cpu")
    estimator.load_model(model_dir / "trees.ubj")
    return estimator


def predict(estimator, bundle, sites):
    matrix = design_matrix(sites, bundle["numeric_features"], bundle["preprocessing"]["scaler"])
    return estimator.predict_proba(matrix)[:, 1]
