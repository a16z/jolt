.PHONY: help bootstrap build-emulator \
        arch-tests-64imac arch-tests-generate arch-tests-run arch-tests-smoke

MAKEFILE_DIR      := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
ARCH_TEST_DIR     := $(MAKEFILE_DIR)/third-party/riscv-arch-test
ARCH_TEST_CONFIG  := $(MAKEFILE_DIR)/tests/arch-tests/jolt/test_config.yaml
# Keep ACT4 build artifacts outside the submodule checkout. ACT4 places them
# under $(ARCH_TEST_WORK)/<name>/, where <name> is `jolt-rv64imac` (the `name`
# field in test_config.yaml). The runner recursively finds *.elf under the
# parent, so it doesn't hard-code the intermediate path.
ARCH_TEST_WORK    := $(MAKEFILE_DIR)/target/arch-tests-work
ARCH_TEST_SKIP    := $(MAKEFILE_DIR)/tests/arch-tests/skip.txt
ARCH_TEST_RUNNER  := $(MAKEFILE_DIR)/tests/arch-tests/run.sh
EMULATOR_BIN      := $(MAKEFILE_DIR)/target/debug/jolt-emu
SMOKE_DIR         := $(MAKEFILE_DIR)/tests/arch-tests/smoke
SMOKE_ELF         := $(SMOKE_DIR)/fail.elf

# Number of parallel jobs for the ACT4 generator. Falls back to 1 when
# neither `nproc` nor `sysctl` is available.
NPROC := $(shell (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 1)

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Install arch-test prerequisites (riscv64 toolchain, sail_riscv_sim)
	./scripts/bootstrap

build-emulator: ## Build the jolt-emu binary in debug mode
	# Debug build is intentional — arch tests are correctness-sensitive and
	# don't benefit from release optimizations.
	cargo build -p tracer --bin jolt-emu

arch-tests-64imac: build-emulator arch-tests-generate arch-tests-run ## Run the ACT4 RV64IMAC suite end-to-end against jolt-emu

arch-tests-generate: ## Generate ACT4 self-checking ELFs via Sail
	# Prepend both mise ($HOME/.local/bin) and sail_riscv_sim (/opt/riscv/bin)
	# to PATH. scripts/bootstrap installs both. The ACT4 top-level Makefile
	# prefers mise for managing the declared Python (uv) + Ruby (bundler)
	# versions in .mise.toml, and shells out to sail_riscv_sim for signature
	# generation.
	#
	# CONFIG_FILES and WORKDIR are passed as make variables. Privileged-mode
	# test generation is disabled via `include_priv_tests: false` in
	# test_config.yaml (spec Non-Goals).
	PATH="$$HOME/.local/bin:/opt/riscv/bin:$$PATH" $(MAKE) -C $(ARCH_TEST_DIR) -j$(NPROC) \
		CONFIG_FILES=$(ARCH_TEST_CONFIG) \
		WORKDIR=$(ARCH_TEST_WORK)

arch-tests-run: ## Run all generated ELFs through jolt-emu and report pass/fail
	$(ARCH_TEST_RUNNER) \
		--emulator $(EMULATOR_BIN) \
		--work-dir $(ARCH_TEST_WORK) \
		--skip-file $(ARCH_TEST_SKIP)

arch-tests-smoke: build-emulator $(SMOKE_ELF) ## Deliberate-failure smoke test — confirms the harness surfaces failures
	@echo "smoke: running deliberate-failure ELF through jolt-emu"
	@if $(EMULATOR_BIN) $(SMOKE_ELF) >/dev/null 2>&1; then \
		echo "smoke FAIL: jolt-emu returned 0 on a deliberate-failure ELF" >&2; \
		echo "the pass/fail plumbing is broken — every real test would silently pass" >&2; \
		exit 1; \
	else \
		rc=$$?; \
		echo "smoke OK: jolt-emu exited $$rc on deliberate-failure ELF"; \
	fi

$(SMOKE_ELF): $(SMOKE_DIR)/fail.S $(MAKEFILE_DIR)/tests/arch-tests/jolt/link.ld
	PATH="/opt/riscv/bin:$$PATH" riscv-none-elf-gcc \
		-march=rv64imac_zicsr -mabi=lp64 \
		-nostdlib -nostartfiles \
		-T $(MAKEFILE_DIR)/tests/arch-tests/jolt/link.ld \
		-o $@ $<
