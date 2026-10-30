.PHONY: install lint test build-data repro demo readme-results clean

install:
	pip install -e ".[dev]"

lint:
	ruff check palimdiff baselines scripts tests
	ruff format --check palimdiff baselines scripts tests

format:
	ruff format palimdiff baselines scripts tests

test:
	pytest -q

PYTHON ?= python3

build-data:
	$(PYTHON) scripts/build_ruler_hqa.py --seed 42 --num 20
	$(PYTHON) scripts/build_retention_probe.py --seed 42 --num 10

repro: build-data test demo
	$(PYTHON) scripts/run_experiments.py --denoiser tiny --seed 42 --rho 0.25
	$(PYTHON) scripts/generate_readme_results.py > results/README_TABLE.md

demo:
	$(PYTHON) scripts/render_demo.py --out docs/assets/demo_mock.gif

readme-results:
	$(PYTHON) scripts/generate_readme_results.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data results/*.json
