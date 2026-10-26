# Mobius: m6A prediction

Mobius predicts the probability of m6A modification at candidate RNA sites from
direct RNA sequencing signals. Use the included model to predict new data, or
fit a new model using your own labelled data.

**Requirements:** Ubuntu 24.04 (Ronin), x86-64, Git and Python 3.12 with `venv`.
Allow 3 GB of free disk space for installation. Dependencies download during
installation; prediction then runs locally on CPU.

## Install and test

```bash
git clone https://github.com/KryanBee/Mobius-m6A.git
cd Mobius-m6A
bash install.sh
./run.sh test
```

The installer handles the Python environment and pinned CPU dependencies; no
environment activation or GPU is needed. The test
runs the real prediction script with our supplied model and unlabelled 12-site
[example](examples/test.json). It prints `PASS` after checking the CSV columns,
site identities/order, finite scores in [0, 1], and agreement with the
[expected predictions](examples/expected_predictions.csv) within 1e-8.
These reference scores check reproducibility; the example contains no ground-truth labels.

The CSV is saved in a new `outputs/example.XXXXXXXX/predictions.csv` each time;
its exact path is printed. You can repeat the test safely. The example takes about 5 seconds on our Ronin machine.
Prediction evaluation requires no training or labels.

## Predict

```bash
./run.sh predict --input dataset.json.gz --output outputs/predictions.csv
```

Choose a new output filename. The comma-separated CSV has this exact header:

```text
transcript_id,transcript_position,score
```

Each row preserves an input site's identifiers, position and order. Scores are
floating-point probabilities in [0, 1]; higher scores indicate stronger m6A evidence.
Run once per dataset.

## Data format

**Signals:** JSON or JSON.gz, one site per line:

```text
{"transcript_id":{"position":{"7-mer":[read_vector, read_vector, ...]}}}
```

Each site needs **at least 20 reads**. Each read has nine finite numeric values:
dwell time, signal standard deviation and signal mean, repeated for the minus-one,
centre and plus-one windows. Dwell times must be positive and standard deviations
nonnegative. Sequences use seven uppercase A/C/G/T bases; the bundled model
supports the course's DRACH-centred candidate sites.

Sites with insufficient coverage, unsupported sequence contexts or undefined
signal statistics raise an error naming the problem. The CSV is published only
after all sites succeed. Sites are never silently filtered. Use the bundled
example for the quick start; other inputs must satisfy this contract.

**Labels for training:** comma-separated files (`.csv`, `data.info` or
`data.info.labelled`) or tab-separated `.tsv` files; gzip is supported.

| Required column | Value |
|---|---|
| `transcript_id` | String matching the signal JSON |
| `transcript_position` | Nonnegative integer matching the JSON position |
| `label` | Finite number in [0, 1]; 0 is negative, every value >0 is positive |

Transcript/position pairs must be unique within each dataset, with exactly
matching signal and label sites. Training requires both classes overall.
`gene_id` is optional and ignored. An optional `source_id` column must match its
`--dataset` source. Supply labels separately.

## Train

```bash
./run.sh train --dataset data0 dataset.json.gz labels.csv --output training-output/my-model
```

This fits the selected recipe on all supplied sites. Repeat
`--dataset SOURCE_ID SIGNAL_JSON LABELS` for additional datasets, using distinct
source IDs. Choose a new output directory. It contains `model/`, the saved recipe,
training histories, input hashes and prediction replay checks. To use the new model:

```bash
./run.sh predict --input new-data.json.gz --output outputs/new-data.csv \
  --model training-output/my-model/model
```

Training uses the bundled fixed settings, seed 1101 and one CPU thread; preserve
dataset and read order for repeatability. It requires more memory and temporary
disk space than prediction. The included model is kept unchanged.
Use `./run.sh train --help` for options including scratch space, or
`./run.sh predict --help` for prediction options.

## Code and reproducibility

| Location | Contents |
|---|---|
| [predict.py](predict.py), [train.py](train.py) | Public scripts, called by `run.sh` |
| [sonic/](sonic/) | Input validation, features, fitting and prediction |
| [model/](model/), [config/final_recipe.json](config/final_recipe.json) | Saved model and fixed training settings |
| [requirements.txt](requirements.txt), [model_checksums.json](model_checksums.json) | Pinned CPU dependencies and model file hashes |
| [examples/](examples/), [tests/](tests/) | Supplied data, expected outputs, verifier and regression tests |

Prediction verifies the model's checkpoint files before use. Additional software
checks, including a small generated training example, can be run with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On our Ronin machine, prediction on 121,838 sites took about 4.6 minutes and under
0.5 GiB peak RAM using one CPU thread. Time and memory depend on reads and hardware.

If installation fails, check Python 3.12/`venv`, free disk space and network access,
then rerun `bash install.sh`. If Python or `venv` is missing, ask your Ronin
administrator to install `python3.12` and `python3.12-venv`. For existing-output
errors, choose a new path. For input errors, check the site and data requirement
named in the message.

Sonic Hedgehog team · NUS DSA4262.
