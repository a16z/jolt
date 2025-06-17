FROM mcr.microsoft.com/devcontainers/base:noble

RUN <<-'EOF' bash
	set -eu -o pipefail
	apt-get update
	apt-get install -y --no-install-recommends ca-certificates pkg-config libssl-dev
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sed 's#/proc/self/exe#\/bin\/sh#g' | sh -s -- -y --no-update-default-toolchain 1>/dev/null
	rm -rf /var/lib/apt/lists/*
EOF
