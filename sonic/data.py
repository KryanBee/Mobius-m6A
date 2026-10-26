"""Strict raw JSON ingestion and portable, label-free read-bag caches."""
from pathlib import Path
from functools import cached_property
import gzip
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
from sonic.common import KEYS, file_info, new_directory, read_json, sha256, utc_now, write_json
from sonic.features import READ_FEATURES, SEQUENCE_ENCODING, encode_sequence, validate_reads

MIN_READS = 20
MAX_LINE_BYTES = 16 * 1024**2


def identifier(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def position_number(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"Invalid transcript position: {value!r}")
    number = int(value)
    if str(number) != value or number > 2**63 - 1:
        raise ValueError(f"Position cannot be preserved as int64: {value!r}")
    return number


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key!r}")
        result[key] = value
    return result


def only_item(mapping, description):
    if not isinstance(mapping, dict) or len(mapping) != 1:
        raise ValueError(f"Expected exactly one {description}")
    return next(iter(mapping.items()))


def raw_bags(path, source_id):
    """Read one complete site per JSON line; never relabel or filter sites."""
    source_id = identifier(source_id)
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    seen = set()
    with opener(path, "rb") as handle:
        line_number = 0
        while True:
            line = handle.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            line_number += 1
            try:
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("JSON line exceeds 16 MiB")
                record = json.loads(line, object_pairs_hook=unique_object)
                transcript, positions = only_item(record, "transcript per line")
                position, sequences = only_item(positions, "position per line")
                sequence, reads = only_item(sequences, "sequence per site")
                key = (identifier(transcript), position_number(position))
                if key in seen:
                    raise ValueError(f"Duplicate JSON site: {key}")
                seen.add(key)
                encode_sequence([sequence])
                if not isinstance(reads, list):
                    raise ValueError("Read vectors must be a list")
                if len(reads) < MIN_READS:
                    raise ValueError(f"Site {key} requires at least 20 reads; received {len(reads)}")
                for index, vector in enumerate(reads, 1):
                    if not isinstance(vector, list) or len(vector) != 9:
                        raise ValueError(f"Read {index} must have nine values")
                    for value in vector:
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError(f"Read {index} has a non-finite/non-numeric value")
                        if isinstance(value, int) and int(float(value)) != value:
                            raise ValueError(f"Read {index}: integer loses precision in float64")
                values = validate_reads(reads)
                from sonic.rich_features import validate_rich_reads
                validate_rich_reads(values)
                yield dict(zip(KEYS, (source_id, *key)), sequence_7mer=sequence,
                           read_count=len(values)), values
            except (ValueError, OverflowError) as error:
                raise ValueError(f"JSON line {line_number}: {error}") from error


class BagChunk:
    """Bounded in-memory bags implementing the same numerical data interface."""
    kind, version, numeric = "bags", "reads9_v1", READ_FEATURES

    def __init__(self, items):
        self.sites = pd.DataFrame([row for row, _ in items])
        validate_keys(self.sites)
        self.sites["_row"] = np.arange(len(items))
        self.arrays = [values for _, values in items]
        self.read_counts = self.sites.read_count.to_numpy()

    @cached_property
    def sequence_features(self):
        return encode_sequence(self.sites.sequence_7mer)

    def read_bag(self, row):
        return self.arrays[int(row)]


def bag_chunks(path, source_id="input", *, chunk_sites=2048, chunk_reads=200000):
    """Bound both sites and reads; an oversized complete site stays intact."""
    if any(type(value) is not int or value <= 0 for value in (chunk_sites, chunk_reads)):
        raise ValueError("Chunk sites and reads must be positive integers")
    pending, reads = [], 0
    for row, values in raw_bags(path, source_id):
        if pending and (len(pending) >= chunk_sites or reads + len(values) > chunk_reads):
            yield BagChunk(pending)
            pending, reads = [], 0
        pending.append((row, values))
        reads += len(values)
    if pending:
        yield BagChunk(pending)


def build_artifact(output, kind, *, json_path=None, source_id=None, feature_version=None):
    """Stream a JSON source into a temporary portable bag cache for training."""
    if kind != "bags" or feature_version not in (None, "reads9_v1") or json_path is None:
        raise ValueError("Training caches require JSON and reads9_v1 bags")
    inputs = [file_info(json_path)]
    started = time.perf_counter()
    with new_directory(output) as temporary:
        rows, offset = [], 0
        with (temporary / "reads.f64").open("wb") as handle:
            for row, values in raw_bags(json_path, source_id):
                rows.append({**row, "offset": offset})
                np.asarray(values, dtype="<f8").tofile(handle)
                offset += len(values)
        if not rows:
            raise ValueError("Input JSON contains no sites")
        pd.DataFrame(rows).to_parquet(temporary / "sites.parquet", index=False)
        if sha256(json_path) != inputs[0]["sha256"]:
            raise ValueError("Input changed during extraction")
        outputs = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in temporary.iterdir()}
        manifest = {"schema_version": 1, "kind": "bags", "feature_version": "reads9_v1",
                    "sequence_encoding": SEQUENCE_ENCODING, "numeric_features": READ_FEATURES,
                    "created_utc": utc_now(), "inputs": inputs, "selection": None,
                    "sites": len(rows), "reads": offset, "dtype": "<f8", "outputs": outputs,
                    "elapsed_seconds": time.perf_counter() - started}
        write_json(temporary / "manifest.json", manifest)
    return manifest


def artifact_identity(manifest):
    """Content identity ignores cache location and extraction timestamp."""
    identity = {k: manifest[k] for k in ("schema_version", "kind", "feature_version",
                                         "sequence_encoding", "numeric_features", "outputs")}
    identity["inputs"] = [item["sha256"] for item in manifest["inputs"]]
    identity["selection"] = manifest["selection"]["sha256"] if manifest["selection"] else None
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_keys(frame, name="table"):
    if frame.empty or not set(KEYS) <= set(frame.columns):
        raise ValueError(f"{name}: empty or missing site keys")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name}: missing or duplicate site keys")
    for column in KEYS[:2]:
        if any(not isinstance(x, str) or not x or x != x.strip() for x in frame[column]):
            raise ValueError(f"{name}: invalid {column}")
    positions = frame[KEYS[2]]
    if not pd.api.types.is_integer_dtype(positions) or positions.lt(0).any():
        raise ValueError(f"{name}: positions must be nonnegative integers")


def read_frame(path):
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",",
                            keep_default_na=False, dtype={k: "str" for k in KEYS[:2]})
    validate_keys(frame, str(path))
    return frame


def exact_join(left, right, name="join"):
    validate_keys(left, name + " left")
    validate_keys(right, name + " right")
    result = left.merge(right, on=KEYS, how="outer", sort=False,
                        validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError(f"{name}: site coverage must match exactly; no silent intersection")
    return result.drop(columns="_merge")


class FeatureData:
    """Validated snapshot of one or more compatible source-specific artifacts.

    Site feature metadata and its positional ``_row`` mapping are immutable for
    this object's lifetime. Select/join separate frames, retaining ``_row``;
    construct a new FeatureData if the underlying feature artifacts change.
    """

    def __init__(self, folders):
        self.manifests, self.arrays, frames = [], [], []
        for i, folder in enumerate(map(Path, folders)):
            manifest = read_json(folder / "manifest.json")
            if manifest["schema_version"] != 1 or manifest["sequence_encoding"] != SEQUENCE_ENCODING:
                raise ValueError("Unsupported feature manifest")
            for name, info in manifest["outputs"].items():
                path = folder / name
                if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                    raise ValueError(f"Artifact checksum mismatch: {path}")
            frame = read_frame(folder / "sites.parquet")
            encode_sequence(frame["sequence_7mer"])
            if len(frame) != manifest["sites"] or frame["read_count"].le(0).any():
                raise ValueError("Invalid site/read counts in artifact")
            if manifest["kind"] == "bags":
                offsets = np.r_[0, frame["read_count"].to_numpy().cumsum()]
                if (manifest["feature_version"] != "reads9_v1"
                        or manifest["numeric_features"] != READ_FEATURES
                        or manifest["dtype"] != "<f8"
                        or offsets[-1] != manifest["reads"]
                        or not np.array_equal(frame["offset"], offsets[:-1])
                        or (folder / "reads.f64").stat().st_size != int(offsets[-1]) * 9 * 8):
                    raise ValueError("Invalid bag offsets/schema/size")
                self.arrays.append(np.memmap(folder / "reads.f64", dtype="<f8", mode="r", shape=(offsets[-1], 9)))
            else:
                raise ValueError("Unsupported artifact kind")
            frame["_artifact"] = i
            frames.append(frame)
            self.manifests.append(manifest)
        if not frames:
            raise ValueError("Supply at least one feature artifact")
        signature = lambda m: (m["kind"], m["feature_version"], m["numeric_features"])
        if any(signature(m) != signature(self.manifests[0]) for m in self.manifests):
            raise ValueError("Incompatible feature artifacts")
        self.sites = pd.concat(frames, ignore_index=True)
        validate_keys(self.sites, "combined features")
        self.sites["_row"] = np.arange(len(self.sites))
        self.kind, self.version, self.numeric = signature(self.manifests[0])
        # Cache only fixed, label-free metadata. Raw reads remain read-only
        # memory maps; learned normalization still belongs to each fitted model.
        self.read_counts = self.sites["read_count"].to_numpy(copy=True)
        self.read_counts.setflags(write=False)
        if self.kind == "bags":
            self._read_offsets = self.sites["offset"].to_numpy(copy=True)
            self._read_artifacts = self.sites["_artifact"].to_numpy(copy=True)
            self._read_offsets.setflags(write=False)
            self._read_artifacts.setflags(write=False)

    @cached_property
    def sequence_features(self):
        """Fixed one-hot encoding in stable _row order, built once when needed."""
        encoded = encode_sequence(self.sites["sequence_7mer"])
        encoded.setflags(write=False)
        return encoded

    def read_bag(self, row):
        row = int(row)
        start, size = int(self._read_offsets[row]), int(self.read_counts[row])
        return self.arrays[int(self._read_artifacts[row])][start:start + size]
