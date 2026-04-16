#!/bin/bash
set -e
set -o pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
[ -f ${DIR}/../.versions ] && source ${DIR}/../.versions

if [[ "$(uname -m)" == "aarch64" ]]; then
    NATIVE_PLATFORM="linux/arm64"
else
    NATIVE_PLATFORM="linux/amd64"
fi

# Default values
REGISTRY=${REGISTRY:-ghcr.io}
BUILDER=${BUILDER:-local-builder}
PLATFORMS=${PLATFORMS:-linux/arm64,linux/amd64}
PUSH=${PUSH:-true}
OS_VERSION=${OS_VERSION:-'noble'}
NAMESPACE=${NAMESPACE:-"a16z"}
PREBUILT_DIR="${DIR}/../prebuilt-files"
RISC_V_DIR="/opt/riscv"

function check_builder() {
    if ! docker buildx ls | grep -q "^${BUILDER}"; then
        local builder_available=false

        if [[ "${BUILDER}" == "local-builder" ]]; then
            echo "Builder '${BUILDER}' does not exist. Creating it automatically..."
            docker buildx create --name local-builder --platform linux/amd64,linux/arm64

            # Check again after creation
            if docker buildx ls | grep -q "^${BUILDER}"; then
                echo "Successfully created builder '${BUILDER}'"
                builder_available=true
            fi
        fi

        if [[ "$builder_available" == "false" ]]; then
            echo "Error: Builder '${BUILDER}' does not exist"
            echo "Available builders:"
            docker buildx ls
            echo ""
            echo "Create local-builder:"
            echo "docker buildx create --name local-builder --platform linux/amd64,linux/arm64"
            exit 1
        fi
    fi
}

function publish_spike() {
    # Create temporary directory for deps context
    DEPS_CONTEXT="$(mktemp -d)"
    trap 'rm -rf "${DEPS_CONTEXT}"' EXIT

    # Create an empty temporary directory as the build context
    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "${TEMP_DIR}"' EXIT

    local BUILD_OUTPUT_OPTION=""
    if [[ "$PUBLISH_LOCAL" == "true" ]]; then
        BUILD_OUTPUT_OPTION="--load"
    else
        BUILD_OUTPUT_OPTION="--push"
    fi

    docker buildx build \
        --builder ${BUILDER} \
        --progress=plain \
        --platform=${PLATFORMS} \
        --build-arg OS_VERSION=${OS_VERSION} \
        --build-arg REGISTRY=${REGISTRY} \
        --build-arg NAMESPACE=${NAMESPACE} \
        --build-context deps="${DEPS_CONTEXT}" \
        --target spike \
        --tag ${REGISTRY}/${NAMESPACE}/spike:1.1.1-${OS_VERSION} \
        -f ${DIR}/Dockerfile.spike ${TEMP_DIR} ${BUILD_OUTPUT_OPTION}
}

function build_spike() {
    local platforms=${1:-$PLATFORMS}
    local target_dir=${2:-$PREBUILT_DIR}
    local OS_VERSION=noble

    mkdir -p "$target_dir"

    # Ensures the generated executable can run across multiple platforms, including Debian Bookworm, Ubuntu Jammy, and Ubuntu Noble, improving portability without requiring additional system dependencies.
    for platform in $platforms; do
        # Create temporary directory for deps context
        DEPS_CONTEXT="$(mktemp -d)"
        trap 'rm -rf "${DEPS_CONTEXT}"' EXIT

        # Create an empty temporary directory as the build context
        TEMP_DIR=$(mktemp -d)
        trap 'rm -rf "${TEMP_DIR}"' EXIT

        docker buildx build \
            --builder ${BUILDER} \
            --progress=plain \
            --platform=${platforms} \
            --build-arg OS_VERSION=${OS_VERSION} \
            --build-arg REGISTRY=${REGISTRY} \
            --build-arg NAMESPACE=${NAMESPACE} \
            --build-context deps="${DEPS_CONTEXT}" \
            --target prebuilt \
            --output type=local,dest="$target_dir" \
            -f ${DIR}/Dockerfile.spike ${TEMP_DIR}
    done
}

function build_prebuilt() {
    build_spike
}

function install_spike() {
    local target_dir=$(mktemp -d)
    trap 'rm -rf "${target_dir}"' EXIT

    build_spike "$NATIVE_PLATFORM" "$target_dir"

    local tar_file=$(find "$target_dir" -name "spike-*.tar.gz" -type f | head -n 1)
    if [[ -z "$tar_file" ]]; then
        echo "Error: No tar.gz file found in $target_dir directory"
        exit 1
    fi

    mkdir -p "$RISC_V_DIR"
    tar -xzf "$tar_file" -C "$RISC_V_DIR"

    echo "Spike installed successfully to $RISC_V_DIR"
}

function install_all() {
    install_spike
}

# Parse command line arguments
COMMAND=""
TARGET=""
PUBLISH_LOCAL=false

function parse_args() {
    if [[ $# -eq 0 ]]; then
        echo "No command specified"
        usage
        exit 1
    fi

    COMMAND="$1"
    shift

    case $COMMAND in
    build)
        if [[ $# -eq 0 ]]; then
            echo "No target specified for build command"
            usage
            exit 1
        fi
        TARGET="$1"
        shift
        ;;
    publish)
        if [[ $# -eq 0 ]]; then
            TARGET="prebuilt"
        else
            TARGET="$1"
            shift
        fi

        # Parse remaining arguments
        while [[ $# -gt 0 ]]; do
            case $1 in
            --builder)
                BUILDER="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --platform)
                PLATFORMS="$2"
                shift 2
                ;;
            --local)
                PUBLISH_LOCAL=true
                PLATFORMS="$NATIVE_PLATFORM"
                PUSH=false
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
            esac
        done
        ;;
    install)
        if [[ $# -eq 0 ]]; then
            TARGET="all"
        else
            TARGET="$1"
            shift
        fi
        ;;
    help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
    esac
}

function usage() {
    cat <<EOF
Usage: $0 <command> [target] [options]

Commands:
  build <target>     Build the specified target locally (target required)
  publish [target]   Build and publish the specified target to registry (defaults to prebuilt)
  install <target>   Install the specified target to system (target required)
  help               Show this help message

Targets:
  spike              Build/install the spike image
  prebuilt           Build the prebuilt files

Options:
  --builder <name>       Specify the builder to use (default: local-builder)
  --registry <registry>  Specify the registry (default: ghcr.io)
  --namespace <name>     Specify the namespace (default: a16z)
  --platform <platforms> Specify platforms (default: linux/arm64,linux/amd64)
  --local               Build only for native platform (implies --platform native)
  --help, -h            Show this help message

Examples:
  $0 build prebuilt
  $0 build spike
  $0 publish                           # Publish prebuilt (default)
  $0 publish spike --builder remote-builder --registry ghcr.io --namespace a16z --platform linux/arm64,linux/amd64
  $0 publish spike --local
  $0 install spike                     # Install spike to $RISC_V_DIR

Environment variables:
  BUILDER: the name of the builder to use (default: local-builder)
  PUSH: whether to push to registry (default: true, only affects publish command)
  PLATFORMS: platforms to build for (default: linux/arm64,linux/amd64)
  REGISTRY: registry to push to (default: ghcr.io)
  NAMESPACE: namespace to use (default: a16z)
  OS_VERSION: OS version to use (default: noble)

EOF
}

function execute() {
    case "$COMMAND:$TARGET" in
    build:spike)
        build_spike
        ;;
    build:prebuilt)
        build_prebuilt
        ;;
    publish:spike)
        publish_spike
        ;;
    install:all)
        install_all
        ;;
    install:spike)
        install_spike
        ;;
    *)
        echo "Error: Unsupported command/target combination '$COMMAND $TARGET'"
        usage
        exit 1
        ;;
    esac
}

parse_args "$@"
check_builder
execute
