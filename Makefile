.PHONY: reproduce render check test

reproduce:  ## execute the notebook (reads data/snapshot/), write results/, refresh README
	cd notebooks && uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute 01_volatility_forecasting.ipynb --output /tmp/01_executed.ipynb --ExecutePreprocessor.timeout=2400
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
