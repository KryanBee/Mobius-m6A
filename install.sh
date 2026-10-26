#!/usr/bin/env bash
# Install the pinned CPU dependencies in this checkout's own environment.
set -euo pipefail

usage() {
    echo "Usage: bash install.sh"
    echo "Create or update this checkout's .venv (Ubuntu x86-64, Python 3.12)."
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
fi

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv_dir="$root_dir/.venv"
environment_python="$venv_dir/bin/python"

fail() {
    printf 'Installation failed: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
    fail "Use an Ubuntu x86-64 machine with Python 3.12."
[[ -f /etc/os-release ]] || fail "Use an Ubuntu machine with Python 3.12."
if ! (source /etc/os-release; [[ "${ID:-}" == ubuntu ]]); then
    fail "Use an Ubuntu machine with Python 3.12."
fi
[[ -f "$root_dir/requirements.txt" ]] || fail "requirements.txt is missing from this checkout."

if [[ -e "$venv_dir" || -L "$venv_dir" ]]; then
    [[ ! -L "$venv_dir" && -d "$venv_dir" && -f "$venv_dir/pyvenv.cfg" && -x "$environment_python" ]] ||
        fail ".venv already exists but is not a local Python environment. Move it aside and rerun this installer."
else
    command -v python3.12 >/dev/null 2>&1 ||
        fail "Python 3.12 is missing. Ask your machine administrator to provide python3.12 and python3.12-venv, then rerun."
    python3.12 -c 'import sys, venv; sys.exit(sys.version_info[:2] != (3, 12))' ||
        fail "Python 3.12 with venv support is required. Ask your machine administrator to provide it, then rerun."
    python3.12 -m venv "$venv_dir" ||
        fail "Could not create .venv. Check available disk space and python3.12-venv support; move an incomplete .venv aside before retrying."
fi

"$environment_python" -I - "$venv_dir" <<'PY' ||
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
valid = (
    sys.version_info[:2] == (3, 12)
    and sys.prefix != sys.base_prefix
    and Path(sys.prefix).resolve() == expected
)
sys.exit(0 if valid else 1)
PY
    fail ".venv must be a local Python 3.12 environment. Move it aside and rerun this installer."

"$environment_python" -I -m pip --isolated --require-virtualenv install -r "$root_dir/requirements.txt" ||
    fail "Dependency installation failed. Check the error above and your internet connection, then rerun bash install.sh."
"$environment_python" -I -m pip --isolated --require-virtualenv check ||
    fail "Dependency verification failed. Check the error above before running the model."
printf '\nInstallation complete. Run the supplied example with:\n  "%s/run.sh" test\n' "$root_dir"
