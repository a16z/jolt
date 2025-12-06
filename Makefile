.PHONY: bootstrap help

MAKEFILE_DIR := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

# Define a general macro for RISCOF test runs
define RISCOF_RUN
	export PATH=$(MAKEFILE_DIR)/target/debug:/opt/riscv/bin:$(PATH); \
	export RUST_BACKTRACE=full; \
	riscof --verbose info run --no-browser --config tests/arch-tests/$(1).ini \
		--suite third-party/riscv-arch-test/riscv-test-suite/ \
		--env third-party/riscv-arch-test/riscv-test-suite/env
endef

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Install required dependencies
	./scripts/bootstrap
	./scripts/apply-patches || true

build-emulator: ## Build the emulator
	# SHOULD NOT USE RELEASE BUILD FOR ARCH TESTS
	cargo build -p tracer --bin jolt-emu

arch-tests-32im: build-emulator
	$(call RISCOF_RUN,jolt-32im)

arch-tests-32imac: build-emulator
	$(call RISCOF_RUN,jolt-32imac)

arch-tests-32gc: build-emulator
	$(call RISCOF_RUN,jolt-32gc)

arch-tests-64im: build-emulator
	$(call RISCOF_RUN,jolt-64im)

arch-tests-64imac: build-emulator
	$(call RISCOF_RUN,jolt-64imac)

arch-tests-64gc: build-emulator
	$(call RISCOF_RUN,jolt-64gc)
