#!/bin/bash
# check_refs.sh — Validate documentation cross-references
# Run before push to catch stale refs
# Usage: ./tools/check_refs.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Checking for stale references ==="

ERRORS=0

# Check for renamed/deleted file refs
echo -n "Checking for fla_replacements.py refs... "
if grep -r "fla_replacements" *.md 2>/dev/null | grep -v "renamed\|was\|from\|→" > /dev/null; then
    echo "FOUND"
    grep -rn "fla_replacements" *.md | grep -v "renamed\|was\|from\|→"
    ERRORS=$((ERRORS + 1))
else
    echo "OK"
fi

# Check for broken markdown links
echo -n "Checking for broken .md links... "
BROKEN=""
for link in $(grep -roh '\[.*\]([^)]*\.md)' *.md 2>/dev/null | sed 's/.*](\([^)]*\)).*/\1/' | sort -u); do
    # Strip anchor
    file=$(echo "$link" | cut -d'#' -f1)
    if [ ! -f "$file" ] && [ ! -f "archive/$file" ]; then
        BROKEN="$BROKEN $file"
    fi
done
if [ -n "$BROKEN" ]; then
    echo "BROKEN:$BROKEN"
    ERRORS=$((ERRORS + 1))
else
    echo "OK"
fi

# Check for refs to non-existent Python files in root
echo -n "Checking Python file refs... "
MISSING=""
for pyfile in $(grep -roh '\[.*\]([^)]*\.py)' *.md 2>/dev/null | sed 's/.*](\([^)]*\)).*/\1/' | sort -u); do
    if [ ! -f "$pyfile" ] && [ ! -f "archive/$pyfile" ]; then
        # Check if it's in a subdir
        base=$(basename "$pyfile")
        if ! find . -name "$base" -type f 2>/dev/null | head -1 | grep -q .; then
            MISSING="$MISSING $pyfile"
        fi
    fi
done
if [ -n "$MISSING" ]; then
    echo "MISSING:$MISSING"
    ERRORS=$((ERRORS + 1))
else
    echo "OK"
fi

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✅ All reference checks passed"
    exit 0
else
    echo "❌ $ERRORS check(s) failed — fix before pushing"
    exit 1
fi
