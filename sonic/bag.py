"""Shared read encoder → within-site mean → optional site MLP → one logit."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from sonic.preprocessing import transform
from sonic.read_normalization import MotifReadScaler, VERSION, SCALING_UNIT, REFERENCE_FILE, fit_and_save


class BagNetwork(nn.Module):
    def __init__(self, hidden_dims, dropout, site_hidden_dims=(), *, tabular_input_dim=0,
                 tabular_hidden_dims=(), read_input_dim=37):
        super().__init__()
        if type(read_input_dim) is not int or read_input_dim not in (27, 37):
            raise ValueError("Read input width must match base28 or DRACH18 sequence encoding")
        layers, width = [], read_input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.tabular_encoder = None
        if tabular_input_dim:
            layers, tabular_width = [], tabular_input_dim
            for hidden in tabular_hidden_dims:
                layers.extend([nn.Linear(tabular_width, hidden), nn.ReLU()])
                tabular_width = hidden
            # Empty dimensions mean direct concatenation of all scaled inputs.
            self.tabular_encoder = nn.Sequential(*layers) if layers else nn.Identity()
            width += tabular_width
        if site_hidden_dims:
            layers = []
            for hidden in site_hidden_dims:
                layers.extend([nn.Linear(width, hidden), nn.ReLU()])
                width = hidden
            self.classifier = nn.Sequential(*layers, nn.Linear(width, 1))
        else:
            # Retain legacy state keys (classifier.weight/bias) and initialization
            # order so original configs and saved checkpoints remain compatible.
            self.classifier = nn.Linear(width, 1)

    def encode_reads(self, reads, site_index, counts):
        """Return only the pooled read branch, before any tabular concatenation."""
        # Parameters are shared across all reads and all sites. The first axis
        # can have any length; index_add groups read representations by site.
        encoded = self.encoder(reads)
        pooled = encoded.new_zeros((len(counts), encoded.shape[1]))
        pooled.index_add_(0, site_index, encoded)
        return pooled / counts[:, None]

    def forward(self, reads, site_index, counts, tabular=None):
        pooled = self.encode_reads(reads, site_index, counts)
        if self.tabular_encoder is not None:
            if tabular is None or len(tabular) != len(counts):
                raise ValueError("Joint network requires one tabular vector per site")
            pooled = torch.cat([pooled, self.tabular_encoder(tabular)], dim=1)
        elif tabular is not None:
            raise ValueError("Read-only network received unexpected tabular inputs")
        return self.classifier(pooled).squeeze(1), pooled


def batches(data, rows, params):
    """Bound both site count and total reads, retaining every complete bag.

    One bag larger than batch_reads gets its own batch. The limit is therefore
    soft for an individual bag; reads are never truncated or resampled.
    """
    pending, count = [], 0
    for row in rows:
        size = int(data.read_counts[int(row)])
        if pending and (len(pending) >= params["batch_sites"] or count + size > params["batch_reads"]):
            yield np.asarray(pending, dtype=int)
            pending, count = [], 0
        pending.append(int(row))
        count += size
    if pending:
        yield np.asarray(pending, dtype=int)


def pack(data, rows, scaler):
    bags = [data.read_bag(row) for row in rows]
    counts = np.asarray([len(b) for b in bags])
    values = np.concatenate(bags)
    numeric = (scaler.transform(data, rows, counts, values) if isinstance(scaler, MotifReadScaler)
               else transform(values, scaler)).astype(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("Scaled read values overflow float32")
    sequence = data.sequence_features[rows]
    features = np.column_stack([numeric, np.repeat(sequence, counts, axis=0)])
    return (torch.from_numpy(features), torch.from_numpy(np.repeat(np.arange(len(rows)), counts)),
            torch.from_numpy(counts.astype(np.float32)))


def predict(model, data, rows, scaler, params, representations=False, site_inputs=None):
    model.eval()
    scores, vectors = [], []
    with torch.no_grad():
        for group in batches(data, rows, params):
            tabular = None if site_inputs is None else torch.from_numpy(site_inputs[group])
            logits, pooled = model(*pack(data, group, scaler), tabular=tabular)
            scores.append(torch.sigmoid(logits).numpy())
            if representations:
                vectors.append(pooled.numpy())
    return np.concatenate(scores), np.concatenate(vectors) if representations else None


def represent_reads(model, data, rows, scaler, params):
    """Frozen read features without evaluating a tabular branch or classifier."""
    model.eval()
    vectors = []
    with torch.no_grad():
        for group in batches(data, rows, params):
            vectors.append(model.encode_reads(*pack(data, group, scaler)).numpy())
    return np.concatenate(vectors)


def fit(data, train, validation, config, weighting, model_dir, history_path, *, fixed_steps=None,
        read_input_dim=37):
    if validation is not None or type(fixed_steps) is not int or fixed_steps <= 0:
        raise ValueError("Submission training requires a positive fixed epoch count, without a stopping set")
    if config.get("read_normalization") != VERSION or read_input_dim != 37:
        raise ValueError("This release uses motif-normalized reads and seven-base encoding")
    params = config["params"]
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    train_rows = train["_row"].to_numpy()
    reference = fit_and_save(data, train, model_dir)
    scaler = MotifReadScaler(data, train_rows, reference)
    model = BagNetwork(params["hidden_dims"], params["dropout"], params.get("site_hidden_dims", ()),
                       read_input_dim=read_input_dim)
    optimiser = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    labels = dict(zip(train_rows, train["label"].astype(int)))
    weights = torch.tensor(weighting["weights"], dtype=torch.float32)
    history = []
    for epoch in range(1, fixed_steps + 1):
        model.train()
        total_loss = 0.0
        for group in batches(data, rng.permutation(train_rows), params):
            targets = torch.tensor([labels[row] for row in group], dtype=torch.float32)
            logits, _ = model(*pack(data, group, scaler))
            # A bag contributes one BCE term regardless of its number of reads.
            losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (losses * weights[targets.long()]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural training loss")
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimiser.step()
            total_loss += float(loss.detach()) * len(group)
        final = epoch == fixed_steps
        if final:
            torch.save(model.state_dict(), model_dir / "weights.pt")
        history.append({"epoch": epoch, "train_weighted_loss": total_loss / len(train),
                        "checkpoint_saved": final})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"epoch={epoch}/{fixed_steps} train_loss={total_loss / len(train):.6f}", flush=True)
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True))
    metadata = {"scaler": None, "scaling_unit": SCALING_UNIT,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "fixed_steps": fixed_steps, "stop_reason": "fixed_steps",
                "motif_reference_file": REFERENCE_FILE}
    return metadata, pd.DataFrame(history)


def load(model_dir, bundle):
    torch.set_num_threads(bundle["threads"])
    params = bundle["params"]
    model = BagNetwork(params["hidden_dims"], params["dropout"], params.get("site_hidden_dims", ()),
                       tabular_input_dim=len(bundle.get("tabular_predictors", [])),
                       tabular_hidden_dims=params.get("tabular_hidden_dims", ()),
                       read_input_dim=bundle.get("preprocessing", {}).get("read_input_dim", 37))
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model
