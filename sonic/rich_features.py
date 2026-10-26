"""Versioned complete-read summaries and fitting-only five-mer references.

readstats_context_v1 implements the 3 October EDA recipe. Changing formulas,
omissions, group membership or error policies requires a new recipe version.
No labels are used by this module. Sequence indicators are appended by tabular.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from sonic.common import KEYS
from sonic.features import READ_FEATURES, encode_sequence, validate_reads

VERSION = "readstats_context_v1"
WINDOWS = ("minus1", "center", "plus1")
MEASURES = ("dwell_time", "signal_sd", "signal_mean")
LOG_CHANNELS = [f"log10_dwell_time_{w}" for w in WINDOWS]
CHANNELS = list(READ_FEATURES) + LOG_CHANNELS
STATS = ("mean", "median", "sd", "p10", "p25", "p75", "p90", "iqr", "mad",
         "tail_asymmetry", "min", "max", "skewness", "excess_kurtosis")
FORMULAS = ("mean(x)", "linear p50", "population SD (ddof=0)", "linear p10",
            "linear p25", "linear p75", "linear p90", "p75-p25",
            "median(abs(x-median(x)))", "(p90+p10-2*median)/(p90-p10)",
            "min(x)", "max(x)", "mean(((x-mean)/SD)^3)", "mean(((x-mean)/SD)^4)-3")
CONTRAST_STATS = ("mean", "median", "sd", "p10", "p90", "iqr")
Z_STATS = ("mean", "median", "sd", "p10", "p90", "fraction_below_m1",
           "fraction_above_p1", "fraction_below_m2", "fraction_above_p2")
PAIRS = [(a, b) for w in range(3) for a, b in
         ((3*w+2, 9+w), (3*w+2, 3*w+1), (3*w+1, 9+w))]
PAIRS += [(a, b) for p in ((9, 10, 11), (1, 4, 7), (2, 5, 8))
          for a, b in ((p[0], p[1]), (p[0], p[2]), (p[1], p[2]))]
OMITTED = [f"motif_z__dwell_time_{w}__fraction_below_m2" for w in WINDOWS]
POLICIES = {"undefined_selected_feature": "error", "unseen_motif": "error",
            "degenerate_reference": "error", "nonpositive_dwell": "error",
            "negative_signal_sd": "error", "imputation": "none"}
SOURCES = {
    "m6anet": "https://www.nature.com/articles/s41592-022-01666-1",
    "nanocompore": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8664944/",
    "xpore_supplement": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41587-021-00949-w/MediaObjects/41587_2021_949_MOESM1_ESM.pdf",
}


def _catalog():
    rows = []

    def add(name, group, formula, origin, context=False, extreme=False):
        selectors = [group]
        if context:
            selectors.append("motif_features")
        if extreme:
            selectors.append("extremes_and_high_moments")
        rows.append({"name": name, "group": group, "selectors": selectors,
                     "formula": formula, "origin": origin, "fitted_reference": context})

    for channel in CHANNELS:
        for stat, formula in zip(STATS, FORMULAS):
            group = ("log_dwell_summaries" if channel.startswith("log10") else
                     "baseline_mean9" if stat == "mean" else "read_distribution")
            add(f"{channel}__{stat}", group, f"{formula} across reads of {channel}",
                "Existing baseline" if group == "baseline_mean9" else
                "Nanocompore-inspired log dwell; our summaries" if group == "log_dwell_summaries" else
                "m6Anet/xPore distribution motivation; our statistics, not their validated recipe",
                extreme=stat in ("min", "max", "skewness", "excess_kurtosis"))
    for a, b in PAIRS:
        add(f"corr__{CHANNELS[a]}__{CHANNELS[b]}", "read_correlations",
            "Pearson correlation across paired complete reads",
            "Nanocompore joint-signal and local-window motivation; our correlations")
    for mode in ("raw", "motif_z"):
        for measure in MEASURES:
            for contrast in ("center_vs_flanks", "right_minus_left"):
                for stat in CONTRAST_STATS:
                    add(f"contrast_{mode}__{measure}__{contrast}__{stat}",
                        "raw_window_contrasts" if mode == "raw" else "motif_window_contrasts",
                        f"{stat} of {mode} per-read " + ("center-(left+right)/2" if contrast == "center_vs_flanks" else "right-left"),
                        "m6Anet/Nanocompore neighboring-window motivation; our contrasts", mode == "motif_z")
    for channel in READ_FEATURES:
        for stat in Z_STATS:
            add(f"motif_z__{channel}__{stat}", "motif_relative",
                f"{stat} of (x-mean_5mer)/SD_5mer; fixed strict +/-1,+/-2 tails",
                "m6Anet normalization; our summaries/cutoffs; mixed background, not unmodified RNA", True)
    for window in WINDOWS:
        add(f"dispersion_ratio__{window}", "dispersion_ratios",
            f"across-read SD(signal_mean_{window}) / mean(signal_sd_{window})",
            "Exploratory ratio; not a calibrated SNR or modification fraction")
    return rows


ALL_CATALOG = _catalog()
ALL_NAMES = [r["name"] for r in ALL_CATALOG]
CATALOG = [r for r in ALL_CATALOG if r["name"] not in OMITTED]
LOCAL_INDICES = np.array([i for i, r in enumerate(ALL_CATALOG) if not r["fitted_reference"]])
LOCAL_NAMES = [ALL_NAMES[i] for i in LOCAL_INDICES]
GROUPS = {r["group"]: sum(x["group"] == r["group"] for x in CATALOG) for r in CATALOG}
SELECTORS = {s: sum(s in r["selectors"] for r in CATALOG)
             for r in CATALOG for s in r["selectors"] if s != "baseline_mean9"}
assert len(ALL_CATALOG) == 342 and len(CATALOG) == 339 and len(LOCAL_NAMES) == 225


def recipe_schema(exclude_groups=()):
    if not isinstance(exclude_groups, (list, tuple)) or any(not isinstance(g, str) for g in exclude_groups):
        raise ValueError("exclude_feature_groups must be a list of group names")
    if len(set(exclude_groups)) != len(exclude_groups) or set(exclude_groups) - SELECTORS.keys():
        raise ValueError(f"Unknown/duplicate feature groups; allowed: {sorted(SELECTORS)}; baseline means are protected")
    chosen = [r for r in CATALOG if not set(r["selectors"]) & set(exclude_groups)]
    return {"schema_version": 1, "recipe": VERSION, "exclude_feature_groups": sorted(exclude_groups),
            "numeric_features": [r["name"] for r in chosen], "features": chosen,
            "group_counts": dict(Counter(r["group"] for r in chosen)), "available_selectors": SELECTORS,
            "fixed_omissions": OMITTED, "omission_reason": "Three constant EDA columns excluded by the frozen v1 definition; not reselected per fold",
            "policies": POLICIES, "sources": SOURCES, "sequence_encoding": "base_onehot_v1"}


class SiteFeatureError(ValueError):
    """Known site-level incompatibility; schema and software errors stay fatal."""

    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


def validate_rich_reads(reads):
    x = validate_reads(reads)
    if (x[:, [0, 3, 6]] <= 0).any():
        raise SiteFeatureError("nonpositive_dwell", "Nonpositive dwell: readstats_context_v1 does not add an epsilon or impute")
    if (x[:, [1, 4, 7]] < 0).any():
        raise SiteFeatureError("negative_signal_sd", "Negative supplied signal SD")
    return x


def summary(x):
    q = np.quantile(x, [.1, .25, .5, .75, .9], axis=0, method="linear")
    mu = x.mean(axis=0)
    centered = x - mu
    sd = np.sqrt(np.mean(centered * centered, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        asym = (q[4] + q[0] - 2*q[2]) / (q[4] - q[0])
        z = centered / sd
        squared = z*z
        skew, kurt = np.mean(squared*z, axis=0), np.mean(squared*squared, axis=0) - 3
    return np.column_stack([mu, q[2], sd, q[0], q[1], q[3], q[4], q[3]-q[1],
                            np.median(np.abs(x-q[2]), axis=0), asym, x.min(axis=0), x.max(axis=0), skew, kurt])


def contrast_values(x):
    return np.column_stack([v for k in range(3) for v in
                            (x[:, 3+k] - (x[:, k]+x[:, 6+k])/2, x[:, 6+k]-x[:, k])])


def contrast_summary(x):
    q = np.quantile(x, [.1, .25, .5, .75, .9], axis=0, method="linear")
    return np.column_stack([x.mean(axis=0), q[2], x.std(axis=0), q[0], q[4], q[3]-q[1]])


def local_vector(reads):
    """225 site-only numbers in LOCAL_NAMES order; undefined statistics stay NaN."""
    x = validate_rich_reads(reads)
    base = np.column_stack([x, np.log10(x[:, [0, 3, 6]])])
    stats = summary(base)
    centered = base - base.mean(axis=0)
    sds = np.sqrt(np.mean(centered**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = [np.mean(centered[:, a]*centered[:, b])/(sds[a]*sds[b]) for a, b in PAIRS]
        ratios = stats[[2, 5, 8], 2] / stats[[1, 4, 7], 0]
    return np.r_[stats.ravel(), corrs, contrast_summary(contrast_values(x)).ravel(), ratios]


def _sites(data, sites):
    """Verify keys/metadata before using the cache's positional row identifiers."""
    if data.kind != "bags" or data.version != "reads9_v1" or data.numeric != list(READ_FEATURES):
        raise ValueError("readstats_context_v1 requires reads9_v1 bags")
    if sites.empty or sites.duplicated(KEYS).any() or "_row" not in sites:
        raise ValueError("Feature transform requires nonempty unique sites with bag row identifiers")
    rows = sites._row.to_numpy()
    if not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= len(data.sites)).any():
        raise ValueError("Invalid bag row identifiers")
    columns = KEYS + ["sequence_7mer", "read_count"]
    if not data.sites.iloc[rows][columns].reset_index(drop=True).equals(sites[columns].reset_index(drop=True)):
        raise ValueError("Bag row identifiers disagree with site keys/sequence/counts")
    encode_sequence(sites.sequence_7mer)
    return rows


def compute_local(data, sites, progress=False):
    rows = _sites(data, sites)
    values = np.empty((len(rows), len(LOCAL_NAMES)), dtype=np.float64)
    for i, row in enumerate(rows):
        try:
            values[i] = local_vector(data.read_bag(row))
        except ValueError as error:
            raise ValueError(f"Site {tuple(sites.iloc[i][KEYS])}: {error}") from error
        if progress and (i+1) % 10000 == 0:
            print(f"site_local_features={i+1}/{len(rows)}", flush=True)
    return pd.concat([sites[KEYS].reset_index(drop=True), pd.DataFrame(values, columns=LOCAL_NAMES)], axis=1)


def align_local(sites, local):
    if list(local.columns) != KEYS + LOCAL_NAMES or local.duplicated(KEYS).any():
        raise ValueError("Local feature cache has wrong columns or duplicate keys")
    # Explicitly preserve requested order; cache may be sorted independently.
    selected = sites[KEYS].merge(local, on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
    if not selected._merge.eq("both").all():
        raise ValueError("Local feature cache is missing requested sites")
    return selected[LOCAL_NAMES].to_numpy(dtype=np.float64)


def _key_digest(sites):
    payload = sites[KEYS].sort_values(KEYS).to_json(orient="records")
    return hashlib.sha256(payload.encode()).hexdigest()


def fit_reference(data, fitting, local=None):
    """Read-weighted moments, pooled across occurrences of each actual five-mer.

    Stable parallel-variance updates avoid subtracting two large squared moments.
    Only the explicitly supplied fitting sites contribute; labels are never read.
    """
    rows = _sites(data, fitting)
    values = align_local(fitting, local) if local is not None else None
    accum = {}
    for i, (row, sequence, size) in enumerate(zip(rows, fitting.sequence_7mer, fitting.read_count)):
        if values is None:
            x = validate_rich_reads(data.read_bag(row))
            mu, sd = x.mean(axis=0), x.std(axis=0)
        else:
            stats = values[i, :126].reshape(9, 14)
            mu, sd = stats[:, 0], stats[:, 2]
        if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd < 0).any():
            raise ValueError("Invalid local moments for the motif reference")
        for w in range(3):
            motif = sequence[w:w+5]
            n, mean, m2, sites = accum.get(motif, (0, np.zeros(3), np.zeros(3), 0))
            delta = mu[w*3:w*3+3] - mean
            total = n + int(size)
            m2 = m2 + int(size)*sd[w*3:w*3+3]**2 + delta**2 * (n*int(size)/total)
            accum[motif] = (total, mean + delta*(int(size)/total), m2, sites+1)
    motifs = {}
    for motif, (count, mu, m2, sites) in sorted(accum.items()):
        sd = np.sqrt(m2/count)
        if not np.isfinite(sd).all() or (sd <= 0).any():
            raise ValueError(f"Degenerate motif reference: {motif}")
        motifs[motif] = {"read_windows": count, "site_windows": sites, "mean": mu.tolist(), "sd": sd.tolist()}
    return {"schema_version": 1, "recipe": VERSION, "weighting": "read_weighted_all_window_occurrences",
            "population": "explicit_fitting_sites_only", "fitting_sites": len(fitting),
            "fitting_reads": int(fitting.read_count.sum()), "fitting_keys_sha256": _key_digest(fitting),
            "motifs": motifs, "policies": POLICIES}


def validate_reference(reference):
    if (not isinstance(reference, dict) or reference.get("schema_version") != 1
            or reference.get("recipe") != VERSION or reference.get("policies") != POLICIES
            or reference.get("population") != "explicit_fitting_sites_only"
            or reference.get("weighting") != "read_weighted_all_window_occurrences"
            or not reference.get("motifs")):
        raise ValueError("Invalid saved motif reference")
    count = 0
    for motif, record in reference["motifs"].items():
        mu, sd = np.asarray(record["mean"]), np.asarray(record["sd"])
        if (len(motif) != 5 or set(motif)-set("ACGT") or mu.shape != (3,) or sd.shape != (3,)
                or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any()
                or type(record["read_windows"]) is not int or record["read_windows"] <= 0):
            raise ValueError(f"Invalid motif reference moments: {motif}")
        count += record["read_windows"]
    if count != 3*reference["fitting_reads"]:
        raise ValueError("Motif reference read counts disagree")


def contextual_vector(x, sequence, local_values, reference):
    missing = [sequence[w:w+5] for w in range(3) if sequence[w:w+5] not in reference["motifs"]]
    if missing:
        raise SiteFeatureError("unseen_motif", f"Unseen motif(s) {missing}; the frozen recipe has no fallback")
    records = [reference["motifs"][sequence[w:w+5]] for w in range(3)]
    mu = np.array([r["mean"] for r in records]).reshape(9)
    sd = np.array([r["sd"] for r in records]).reshape(9)
    z = (x-mu)/sd
    stats = local_values[:126].reshape(9, 14)
    locations = (stats[:, [0, 1, 2, 3, 6]]-mu[:, None])/sd[:, None]
    locations[:, 2] = stats[:, 2]/sd
    tails = np.column_stack([(z < -1).mean(axis=0), (z > 1).mean(axis=0),
                             (z < -2).mean(axis=0), (z > 2).mean(axis=0)])
    return np.r_[contrast_summary(contrast_values(z)).ravel(), np.column_stack([locations, tails]).ravel()]


def transform_sites(data, sites, schema, reference=None, local=None, progress=False, *, report_invalid=False):
    rows = _sites(data, sites)
    if schema != recipe_schema(schema["exclude_feature_groups"]):
        raise ValueError("Feature recipe schema has changed")
    need_context = any(r["fitted_reference"] for r in schema["features"])
    if need_context:
        validate_reference(reference)
    values = align_local(sites, local) if local is not None else None
    indices = [ALL_NAMES.index(name) for name in schema["numeric_features"]]
    output = np.full((len(sites), len(indices)), np.nan, dtype=np.float64)
    valid = np.ones(len(sites), dtype=bool)
    reasons, details = [""] * len(sites), [""] * len(sites)
    for i, (row, sequence) in enumerate(zip(rows, sites.sequence_7mer)):
        try:
            x = validate_rich_reads(data.read_bag(row)) if need_context or values is None else None
            v = local_vector(x) if values is None else values[i]
            full = np.full(342, np.nan)
            full[LOCAL_INDICES] = v
            output[i] = full[indices]
            if need_context:
                full[222:339] = contextual_vector(x, sequence, v, reference)
            selected = full[indices]
            output[i] = selected
            if not np.isfinite(selected).all():
                bad = [schema["numeric_features"][j] for j in np.flatnonzero(~np.isfinite(selected))]
                raise SiteFeatureError("undefined_features", f"Undefined selected features (no imputation): {bad[:8]}")
        except SiteFeatureError as error:
            if not report_invalid:
                raise ValueError(f"Site {tuple(sites.iloc[i][KEYS])}: {error}") from error
            valid[i], reasons[i], details[i] = False, error.reason, str(error)
        except ValueError as error:
            raise ValueError(f"Site {tuple(sites.iloc[i][KEYS])}: {error}") from error
        if progress and (i+1) % 10000 == 0:
            print(f"fold_features={i+1}/{len(sites)}", flush=True)
    if report_invalid:
        # Undefined exported statistics are missing values, never usable inputs.
        output[~np.isfinite(output)] = np.nan
    frame = pd.concat([sites.reset_index(drop=True),
                       pd.DataFrame(output, columns=schema["numeric_features"])], axis=1)
    if report_invalid:
        qc = sites[KEYS + ["read_count"]].reset_index(drop=True).copy()
        qc["scorable"], qc["reason"], qc["detail"] = valid, reasons, details
        return frame, qc
    return frame


def transform_identity(schema, reference):
    return hashlib.sha256(json.dumps([schema, reference], sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class PreparedRichData:
    """In-memory fold matrix; only the producing reference may reuse its columns."""

    sites: pd.DataFrame
    schema: dict
    reference: dict | None
    kind: str = "summaries"
    version: str = VERSION

    @property
    def numeric(self):
        return self.schema["numeric_features"]

    @property
    def transform_id(self):
        return transform_identity(self.schema, self.reference)
