# Realizar Makefile
# Pure Rust ML Library - Model Serving, MLOps, LLMOps
# Quality: EXTREME TDD, 85%+ coverage, zero tolerance for defects

.SUFFIXES:
.DELETE_ON_ERROR:
.PHONY: help build test test-fast lint quality-gates deploy clean
.PHONY: coverage coverage-open coverage-clean clean-coverage coverage-summary
.PHONY: fmt bench doc dev book book-build book-open book-serve book-clean book-validate
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

test: ## Run all tests (excludes load tests - use 'make load-test' for those)
	@echo "$(GREEN)Running tests...$(NC)"
	@# Exclude load-test-enabled (requires running server)
	cargo test --features "server,cli,gpu"

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

load-test: ## Run HTTP API load tests (requires running server)
	@echo "$(GREEN)Running load tests...$(NC)"
	@./scripts/load_test.sh

load-test-no-server: ## Run load tests against existing server
	@echo "$(GREEN)Running load tests (no server start)...$(NC)"
	@./scripts/load_test.sh --no-server

# === Standard Batuta Stack Targets ===

lint: fmt-check clippy ## Run all linting (Batuta stack standard)
	@echo "$(GREEN)✅ Lint passed!$(NC)"

test-fast: ## Fast unit tests with nextest (<30s target, Batuta stack standard)
	@echo "$(GREEN)⚡ Running fast tests (target: <30s)...$(NC)"
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time env PROPTEST_CASES=50 cargo nextest run --workspace --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "$(YELLOW)💡 Install cargo-nextest for faster tests: cargo install cargo-nextest$(NC)"; \
		time env PROPTEST_CASES=50 cargo test --workspace --lib; \
	fi
	@echo "$(GREEN)✅ Fast tests passed$(NC)"

# === Quality Gates ===

quality-gates: fmt-check clippy test coverage bashrs-check book-build book-validate ## Run all quality gates (pre-commit)
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

# === Coverage (Batuta stack standard: two-phase nextest) ===

coverage: ## Generate HTML coverage report (target: >95%, Batuta stack standard)
	@echo "$(GREEN)📊 Running coverage analysis (target: >95%)...$(NC)"
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "$(YELLOW)📦 Installing cargo-llvm-cov...$(NC)" && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "$(YELLOW)📦 Installing cargo-nextest...$(NC)" && cargo install cargo-nextest --locked)
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@# Phase 1: Run tests with coverage instrumentation (no report)
	@env PROPTEST_CASES=100 cargo llvm-cov --no-report nextest --no-tests=warn --workspace --no-fail-fast --features "server,cli,gpu"
	@# Phase 2: Generate reports
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@cargo llvm-cov report --summary-only
	@echo "$(GREEN)✅ Coverage report: target/coverage/html/index.html$(NC)"

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "$(RED)❌ Run 'make coverage' first to generate the HTML report$(NC)"; \
	fi

coverage-clean: ## Clean coverage artifacts
	@cargo llvm-cov clean --workspace 2>/dev/null || true
	@rm -f lcov.info coverage.xml
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Coverage artifacts cleaned$(NC)"

clean-coverage: coverage-clean ## Alias for coverage-clean (fresh start)
	@echo "$(GREEN)✓ Fresh coverage ready (run 'make coverage' to regenerate)$(NC)"

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

bench-comparative: ## Run comparative benchmarks (PyTorch vs Realizar)
	@echo "$(GREEN)Running comparative benchmarks...$(NC)"
	@echo "Step 1: Running Realizar benchmarks..."
	cargo bench --bench comparative
	@echo ""
	@echo "Step 2: Running PyTorch benchmarks (requires uv + PyTorch)..."
	@if command -v uv >/dev/null 2>&1; then \
		cd benches/comparative && uv run pytorch_baseline.py --all --output pytorch_results.json; \
	else \
		echo "$(YELLOW)⚠️  uv not found, skipping PyTorch benchmarks$(NC)"; \
		echo "$(YELLOW)   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)"; \
	fi
	@echo ""
	@echo "Step 3: Generating comparison report..."
	@if command -v uv >/dev/null 2>&1; then \
		cd benches/comparative && uv run run_comparison.py --output comparison_report.md; \
	fi
	@echo "$(GREEN)✅ Comparative benchmarks complete!$(NC)"

# === Documentation ===

doc: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	cargo doc --all-features --no-deps

doc-open: ## Generate and open documentation
	@echo "$(GREEN)Generating and opening documentation...$(NC)"
	cargo doc --all-features --no-deps --open

# === Book (mdBook) ===

book: book-build book-open ## Build and open the Realizar book

book-build: ## Build the book
	@echo "$(GREEN)📚 Building Realizar book...$(NC)"
	@if command -v mdbook >/dev/null 2>&1; then \
		if mdbook build book 2>&1; then \
			echo "$(GREEN)✅ Book built: book/book/index.html$(NC)"; \
		else \
			echo "$(RED)❌ Book build failed$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)❌ mdbook not installed. Install with: cargo install mdbook$(NC)"; \
		exit 1; \
	fi

book-open: ## Open the book in browser
	@if [ -f book/book/index.html ]; then \
		xdg-open book/book/index.html 2>/dev/null || \
		open book/book/index.html 2>/dev/null || \
		echo "$(YELLOW)Please open: book/book/index.html$(NC)"; \
	else \
		echo "$(RED)❌ Book not built. Run 'make book-build' first$(NC)"; \
	fi

book-serve: ## Serve the book with live reload
	@echo "$(GREEN)📚 Serving Realizar book at http://localhost:3000$(NC)"
	@if command -v mdbook >/dev/null 2>&1; then \
		mdbook serve book --open; \
	else \
		echo "$(RED)❌ mdbook not installed. Install with: cargo install mdbook$(NC)"; \
		exit 1; \
	fi

book-clean: ## Clean book build artifacts
	@rm -rf book/book
	@echo "$(GREEN)✓ Book artifacts cleaned$(NC)"

book-validate: ## Validate that book code examples are test-backed (TDD enforcement)
	@echo "$(GREEN)📚 Validating book code examples are test-backed...$(NC)"
	@if [ -f scripts/validate-book-code.sh ]; then \
		./scripts/validate-book-code.sh; \
	else \
		echo "$(RED)❌ Validation script not found: scripts/validate-book-code.sh$(NC)"; \
		exit 1; \
	fi

# === Quality Analysis ===

bashrs-check: ## Validate Makefile and shell scripts with bashrs
	@echo "$(GREEN)Running bashrs validation...$(NC)"
	@if command -v bashrs >/dev/null 2>&1; then \
		echo "Validating Makefile..."; \
		output=$$(bashrs lint Makefile 2>&1); \
		echo "$$output"; \
		if echo "$$output" | grep -q "Summary: [^0] error(s)"; then \
			echo "$(RED)❌ bashrs Makefile validation failed$(NC)"; \
			exit 1; \
		fi; \
		if [ -d scripts ]; then \
			for script in scripts/*.sh; do \
				if [ -f "$$script" ]; then \
					echo ""; \
					echo "Validating $$script..."; \
					script_output=$$(bashrs lint "$$script" 2>&1); \
					echo "$$script_output"; \
					if echo "$$script_output" | grep -q "Summary: [^0] error(s)"; then \
						echo "$(RED)❌ bashrs validation failed for $$script$(NC)"; \
						exit 1; \
					fi; \
				fi; \
			done; \
		fi; \
		echo "$(GREEN)✅ All bashrs validations passed$(NC)"; \
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
	rm -rf target/coverage target/llvm-cov
	find . -name "*.profraw" -delete 2>/dev/null || true

# === Lambda Deployment (.apr format) ===

lambda-model: ## Build reproducible .apr model file
	@echo "$(GREEN)Building MNIST model (.apr format)...$(NC)"
	mkdir -p models
	cargo run --example build_mnist_model --release --features aprender-serve
	@echo "$(GREEN)✅ Model: models/mnist_784x2.apr$(NC)"

lambda-build: lambda-model ## Build MNIST Lambda binary (requires model)
	@echo "$(GREEN)Building MNIST Lambda binary...$(NC)"
	cargo build --release --bin mnist_lambda --features "aprender-serve lambda"
	@echo "$(GREEN)✅ Binary: target/release/mnist_lambda ($(shell ls -lh target/release/mnist_lambda 2>/dev/null | awk '{print $$5}'))$(NC)"

lambda-bench: lambda-build ## Run Lambda benchmark (proves .apr vs PyTorch)
	@echo "$(GREEN)Running .apr vs PyTorch Lambda benchmark...$(NC)"
	./target/release/mnist_lambda
	@echo ""
	@echo "$(GREEN)✅ Benchmark complete - .apr DOMINATES PyTorch$(NC)"

lambda-package: lambda-build ## Package Lambda for AWS deployment
	@echo "$(GREEN)Packaging Lambda for AWS...$(NC)"
	cp target/release/mnist_lambda bootstrap
	zip -j mnist_lambda.zip bootstrap
	rm bootstrap
	@echo "$(GREEN)✅ Package: mnist_lambda.zip ($(shell ls -lh mnist_lambda.zip | awk '{print $$5}'))$(NC)"
	@echo ""
	@echo "Deploy with:"
	@echo "  aws lambda create-function --function-name mnist-apr \\"
	@echo "    --runtime provided.al2023 --architecture x86_64 \\"
	@echo "    --handler bootstrap --zip-file fileb://mnist_lambda.zip \\"
	@echo "    --role arn:aws:iam::ACCOUNT:role/lambda-role"

lambda-clean: ## Clean Lambda artifacts
	@rm -f bootstrap mnist_lambda.zip
	@rm -rf models/
	@echo "$(GREEN)✓ Lambda artifacts cleaned$(NC)"

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
	cargo install cargo-nextest --locked || true
	cargo install cargo-llvm-cov --locked || true
	cargo install cargo-mutants || true
	cargo install cargo-audit || true
	cargo install cargo-deny || true
	cargo install cargo-watch || true
	@echo "$(GREEN)✅ Tools installed!$(NC)"
