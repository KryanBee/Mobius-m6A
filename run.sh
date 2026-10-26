#!/usr/bin/env bash
# Run public commands with this checkout's installed Python environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./run.sh COMMAND [arguments]

  test       Run, verify and save the supplied example under outputs/
  predict    Predict scores from a signal JSON
  train      Fit a new model from signal JSON and label tables

Run ./run.sh predict --help or ./run.sh train --help for command options.
Install dependencies first with bash install.sh.
HELP
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
command_name=$1
shift
case "$command_name" in
    --help|-h|help) usage; exit 0 ;;
    test)
        [[ $# -eq 0 ]] || { echo "The test command takes no arguments." >&2; exit 2; }
        ;;
    predict|train) ;;
    *) printf 'Unknown command: %s\n' "$command_name" >&2; usage >&2; exit 2 ;;
esac

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
environment_python="$root_dir/.venv/bin/python"
if [[ ! -x "$environment_python" ]]; then
    printf 'Environment missing. Install dependencies with:\n  bash "%s/install.sh"\n' "$root_dir" >&2
    exit 1
fi

case "$command_name" in
    predict|train)
        exec "$environment_python" "$root_dir/$command_name.py" "$@"
        ;;
    test)
        if ! mkdir -p -- "$root_dir/outputs"; then
            printf 'Example failed: cannot create output directory %s\n' "$root_dir/outputs" >&2
            exit 1
        fi
        if ! example_dir=$(mktemp -d -- "$root_dir/outputs/example.XXXXXXXX"); then
            printf 'Example failed: cannot create a new example directory in %s\n' "$root_dir/outputs" >&2
            exit 1
        fi
        trap 'rm -rf -- "$example_dir"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        "$environment_python" "$root_dir/predict.py" \
            --input "$root_dir/examples/test.json" --output "$example_dir/predictions.csv"
        "$environment_python" "$root_dir/examples/verify.py" "$example_dir/predictions.csv"
        trap - EXIT INT TERM
        printf 'Verified example CSV: %s\n' "$example_dir/predictions.csv"
        ;;
esac
