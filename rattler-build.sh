#!/bin/bash

set -euo pipefail

declare CONDA_CHANNEL
SRCDIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
CONDA_CHANNEL="$SRCDIR/build"

export PLATFORM="noarch"
export GIT_REPO_URL="https://github.com/lutrarutra/deconv"
export GIT_TAG=$(git ls-remote --tags --refs $GIT_REPO_URL | awk -F/ '{print $NF}' | sed 's/^v//' | sort -V | tail -n 1)
export GIT_COMMIT_HASH=$(git ls-remote $GIT_REPO_URL HEAD | awk '{print $1}')

mkdir -p "$CONDA_CHANNEL/$PLATFORM"
MAX_BN=$(find "$CONDA_CHANNEL/$PLATFORM" -name "deconv-$GIT_TAG-*.tar.bz2" 2>/dev/null | sed -E 's/.*_([0-9]+)\.tar.bz2$/\1/' | sort -n | tail -1)

if [ -z "$MAX_BN" ]; then
    export NEW_BUILD_NUMBER=0
else
    export NEW_BUILD_NUMBER=$((MAX_BN + 1))
fi

echo "Next build number for $GIT_TAG: $NEW_BUILD_NUMBER"

rattler-build build \
  --channel="file://${CONDA_CHANNEL}" \
  --channel='https://conda.anaconda.org/conda-forge' \
  --channel='https://conda.anaconda.org/bioconda' \
  --color-build-log \
  --package-format='tar-bz2' \
  --log-style='plain' \
  --output-dir="${CONDA_CHANNEL}" \
  --recipe "recipe.yaml"

chmod --recursive 'g+rX' "${CONDA_CHANNEL}"

echo 'All done.'\