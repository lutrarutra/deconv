#!/bin/bash

set -euo pipefail

SRCDIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd $SRCDIR

if [ -n "$(git status --porcelain)" ]; then
    echo "📦 Uncommitted changes detected."

    # 2. Get the latest tag to use as a default suggestion
    # If no tags exist, it defaults to v0.0.0
    latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")

    # 3. Ask for the new tag
    read -p "Enter new tag name (default: $latest_tag): " new_tag
    new_tag=${new_tag:-$latest_tag} # Use latest_tag if input is empty

    # 4. Ask for the commit message
    read -p "Enter commit message: " commit_msg
    if [ -z "$commit_msg" ]; then
        echo "❌ Error: Commit message cannot be empty."
        exit 1
    fi

    # 5. Add, Commit, Tag, and Push
    echo "🚀 Processing updates..."
    
    # Git add and commit (the -a includes all tracked changes)
    git add .
    git commit -m "$commit_msg"

    # 2. Get the current branch name
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    # 3. Handle the Tagging Logic
    if [ "$new_tag" != "$latest_tag" ]; then
        # SCENARIO A: User provided a brand new tag name
        echo "🏷️  Creating new tag: $new_tag"
        git tag "$new_tag"
        
        echo "🚀 Pushing code and new tag..."
        git push origin "$current_branch"
        git push origin "$new_tag"

    elif git rev-parse "$new_tag" >/dev/null 2>&1; then
        # SCENARIO B: User kept the same tag name, and it already exists
        read -p "Tag '$new_tag' already exists. Move it to this new commit? (y/N): " move_tag
        
        if [[ "$move_tag" =~ ^[yY]$ ]]; then
            echo "🏷️  Updating existing tag: $new_tag"
            git tag -f "$new_tag"
            
            echo "🚀 Pushing code and updated tag..."
            git push origin "$current_branch"
            git push origin "$new_tag" --force
        else
            echo "🚀 Pushing code only (skipping tag update)..."
            git push origin "$current_branch"
        fi
    else
        # SCENARIO C: First time creating a tag (e.g., repository was v0.0.0)
        git tag "$new_tag"
        git push origin "$current_branch"
        git push origin "$new_tag"
    fi

    echo "✅ Successfully committed, tagged as $new_tag, and pushed to $current_branch."
else
    echo "✅ Nothing to commit, working tree clean."
fi

uv build
uv publish --repository pypi

./rattler-build.sh

read -p "Release to Anaconda? (y/N): " release_to_anaconda

if [[ "$release_to_anaconda" =~ ^[yY]$ ]]; then
    anaconda upload ./build/noarch/deconv-$GIT_TAG-$NEW_BUILD_NUMBER.tar.bz2
fi