#!/usr/bin/env bash
set -uo pipefail
command -v gitscribe >/dev/null 2>&1 || exit 0
gitscribe post-merge --ff="$1"
exit 0
