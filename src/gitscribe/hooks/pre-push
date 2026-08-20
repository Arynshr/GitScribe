#!/usr/bin/env bash
set -uo pipefail
command -v gitscribe >/dev/null 2>&1 || exit 0
has_valid_ref=false
while read -r local_ref local_sha remote_ref remote_sha; do
    [ "$local_sha" != "0000000000000000000000000000000000000000" ] && has_valid_ref=true
done
[ "$has_valid_ref" = false ] && exit 0
gitscribe pre-push
exit $?
