#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-o <openmp|threads|cuda>] [-c <custom_kokkos_path>] [-h]"
    echo "  -o <backend>         Specify the backend (openmp, threads, cuda)"
    echo "  -c <path>            Specify a custom Kokkos installation path"
    echo "  -h                   Display this help message"
    exit 1
}

# Default values
BACKEND=""
CUSTOM_KOKKOS_PATH=""

# Parse command line arguments
while getopts ":o:c:h" opt; do
    case ${opt} in
        o )
            BACKEND=$OPTARG
            ;;
        c )
            CUSTOM_KOKKOS_PATH=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument." 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Validate backend
if [[ "$BACKEND" != "openmp" && "$BACKEND" != "threads" && "$BACKEND" != "cuda" ]]; then
    echo "Error: Invalid backend specified. Supported backends are openmp, threads, cuda."
    usage
fi

# Set build directory based on backend
BUILD_DIR="build_${BACKEND}"

# Generate CMake command
CMAKE_CMD="cmake -B $BUILD_DIR"

# Add Kokkos options based on backend
case "$BACKEND" in
    openmp )
        CMAKE_CMD+=" -DKokkos_ENABLE_OPENMP=ON"
        ;;
    threads )
        CMAKE_CMD+=" -DKokkos_ENABLE_THREADS=ON"
        ;;
    cuda )
        CMAKE_CMD+=" -DKokkos_ENABLE_CUDA=ON -DKOKKOS_ARCH_AUTODETECT=ON"
        ;;
esac

# Add custom Kokkos path if specified
if [[ -n "$CUSTOM_KOKKOS_PATH" ]]; then
    CMAKE_CMD+=" -DKokkos_ROOT=$CUSTOM_KOKKOS_PATH"
fi
CMAKE_CMD+=" -DKokkos_ENABLE_DEBUG=ON"
# Print CMake command for debugging
echo "Running CMake command: $CMAKE_CMD"

# Run CMake
$CMAKE_CMD || { echo "CMake configuration failed"; exit 1; }

# Build project
echo "Building project in $BUILD_DIR..."
cmake --build $BUILD_DIR || { echo "Build failed"; exit 1; }

echo "Build completed successfully in $BUILD_DIR"



