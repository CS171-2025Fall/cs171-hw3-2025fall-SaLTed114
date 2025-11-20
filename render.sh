#!/bin/bash

# -----------------------------------------
# Usage: ./run.sh data/scene.json
# -----------------------------------------

if [ $# -ne 1 ]; then
    echo "Usage: $0 <json_scene_file>"
    exit 1
fi

scene="$1"

if [ ! -f "$scene" ]; then
    echo "❌ File not found: $scene"
    exit 1
fi

# Extract base filename (no path)
base=$(basename "$scene" .json)
output_local="${base}.exr"         # renderer output
output_data="data/${base}.exr"      # display path

# -----------------------------------------
# Build the project
# -----------------------------------------
echo "Building the project..."
if ! cmake --build build; then
    echo "❌ Build failed."
    exit 1
fi

# Renderer executable
RENDERER="./build/src/renderer"

if [ ! -x "$RENDERER" ]; then
    echo "❌ Renderer not found: $RENDERER"
    exit 1
fi

# -----------------------------------------
# Run the renderer (STRICT FORMAT)
# -----------------------------------------
echo "----------------------------------"
echo "Rendering:"
echo "  Scene : $scene"
echo "  Output(local): $output_local"
echo "  Will display: $output_data"
echo "Command:"
echo "  $RENDERER $scene -o $output_local"
echo "----------------------------------"

if ! "$RENDERER" "$scene" -o "$output_local"; then
    echo "❌ Rendering failed!"
    exit 1
fi

# -----------------------------------------
# Copy to data/ for display
# -----------------------------------------
# cp "$output_local" "$output_data"

# -----------------------------------------
# Optional display
# -----------------------------------------
if command -v display >/dev/null 2>&1; then
    echo "Displaying ${output_data} ..."
    display "$output_data"
else
    echo "ℹ️  'display' not available — skipping preview."
fi
