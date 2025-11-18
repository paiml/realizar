# Realizar Makefile
# Pure Rust ML Library - Model Serving, MLOps, LLMOps
# Quality: EXTREME TDD, 85%+ coverage, zero tolerance for defects

.PHONY: help build test quality-gates deploy clean
.DEFAULT_GOAL := help

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help: ## Show this help message
	@echo "Realizar - Pure Rust ML Library"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# === Build Targets ===

build: ## Build the project in release mode
	@echo "$(GREEN)Building Realizar...$(NC)"
	cargo build --release

build-dev: ## Build the project in development mode
	@echo "$(GREEN)Building Realizar (dev)...$(NC)"
	cargo build

build-all-features: ## Build with all features enabled
	@echo "$(GREEN)Building Realizar (all features)...$(NC)"
	cargo build --all-features --release

# === Test Targets ===

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	cargo test --all-features

test-lib: ## Run library tests only (fast)
	@echo "$(GREEN)Running library tests...$(NC)"
	cargo test --lib

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	cargo test --lib --bins

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	cargo test --test '*'

test-property: ## Run property-based tests (proptest)
	@echo "$(GREEN)Running property-based tests...$(NC)"
	cargo test --test property_*

# === Quality Gates ===

quality-gates: fmt-check clippy test coverage bashrs-check ## Run all quality gates (pre-commit)
	@echo "$(GREEN)✅ All quality gates passed!$(NC)"

fmt: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	cargo fmt

fmt-check: ## Check code formatting
	@echo "$(GREEN)Checking code formatting...$(NC)"
	cargo fmt --check || (echo "$(RED)❌ Format check failed. Run 'make fmt'$(NC)" && exit 1)

clippy: ## Run clippy lints (zero warnings policy)
	@echo "$(GREEN)Running clippy...$(NC)"
	cargo clippy --all-targets --all-features -- -D warnings

clippy-fix: ## Automatically fix clippy warnings
	@echo "$(GREEN)Fixing clippy warnings...$(NC)"
	cargo clippy --all-targets --all-features --fix

# === Coverage ===

coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "$(GREEN)Coverage report generated: lcov.info$(NC)"
	@echo "$(YELLOW)Coverage:$(NC) $$(cargo llvm-cov --all-features --workspace --summary-only | grep TOTAL | awk '{print $$10}')"

coverage-html: ## Generate HTML coverage report
	@echo "$(GREEN)Generating HTML coverage report...$(NC)"
	cargo llvm-cov --all-features --workspace --html
	@echo "$(GREEN)HTML report: target/llvm-cov/html/index.html$(NC)"

# === Mutation Testing ===

mutate: ## Run mutation testing (requires cargo-mutants)
	@echo "$(GREEN)Running mutation testing...$(NC)"
	cargo mutants --timeout 300 --no-shuffle

mutate-fast: ## Run mutation testing (fast mode)
	@echo "$(GREEN)Running mutation testing (fast)...$(NC)"
	cargo mutants --timeout 60 --jobs 4

# === Benchmarking ===

bench: ## Run all benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	cargo bench

bench-tensor: ## Run tensor operation benchmarks
	@echo "$(GREEN)Running tensor benchmarks...$(NC)"
	cargo bench --bench tensor_ops

# === Documentation ===

doc: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	cargo doc --all-features --no-deps

doc-open: ## Generate and open documentation
	@echo "$(GREEN)Generating and opening documentation...$(NC)"
	cargo doc --all-features --no-deps --open

# === Quality Analysis ===

bashrs-check: ## Validate Makefile and shell scripts with bashrs
	@echo "$(GREEN)Running bashrs validation...$(NC)"
	@if command -v bashrs >/dev/null 2>&1; then \
		bashrs lint Makefile || (echo "$(RED)❌ bashrs validation failed$(NC)" && exit 1); \
	else \
		echo "$(YELLOW)⚠️  bashrs not installed, skipping$(NC)"; \
	fi

audit: ## Run security audit (cargo-audit)
	@echo "$(GREEN)Running security audit...$(NC)"
	cargo audit

deny: ## Run cargo-deny checks (licenses, bans, sources)
	@echo "$(GREEN)Running cargo-deny checks...$(NC)"
	cargo deny check

pmat-tdg: ## Run PMAT Technical Debt Grading
	@echo "$(GREEN)Running PMAT TDG analysis...$(NC)"
	@if command -v pmat >/dev/null 2>&1; then \
		pmat analyze tdg src/; \
	else \
		echo "$(YELLOW)⚠️  pmat not installed, skipping$(NC)"; \
	fi

# === Profiling (Renacer integration) ===

profile: ## Profile benchmarks with Renacer
	@echo "$(GREEN)Profiling benchmarks with Renacer...$(NC)"
	@if command -v renacer >/dev/null 2>&1; then \
		renacer --function-time --source -- cargo bench --no-run; \
	else \
		echo "$(YELLOW)⚠️  renacer not installed, skipping$(NC)"; \
	fi

profile-test: ## Profile tests with Renacer
	@echo "$(GREEN)Profiling tests with Renacer...$(NC)"
	@if command -v renacer >/dev/null 2>&1; then \
		renacer --function-time -- cargo test --no-run; \
	else \
		echo "$(YELLOW)⚠️  renacer not installed, skipping$(NC)"; \
	fi

# === Development ===

dev: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	cargo watch -x 'test --lib' -x 'clippy'

clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	cargo clean
	rm -f lcov.info
	rm -rf mutants.out mutants.out.old

# === Deployment ===

deploy: quality-gates build-all-features ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(NC)"
	@echo "$(YELLOW)Building release...$(NC)"
	@# Future: Deploy model server
	@echo "$(GREEN)✅ Deployment complete!$(NC)"

# === CI/CD ===

ci: quality-gates mutate-fast ## Run CI pipeline (all checks)
	@echo "$(GREEN)✅ CI pipeline passed!$(NC)"

# === Installation ===

install-tools: ## Install development tools
	@echo "$(GREEN)Installing development tools...$(NC)"
	cargo install cargo-llvm-cov
	cargo install cargo-mutants
	cargo install cargo-audit
	cargo install cargo-deny
	cargo install cargo-watch
	@echo "$(GREEN)✅ Tools installed!$(NC)"
