src := reverse_feature_selection
other := feature_selection_benchmark
tests := tests

check:
	poetry run black $(src) $(other) $(tests) --check --diff
	poetry run mypy --install-types --non-interactive --check-untyped-defs $(src) $(other) $(tests)
	poetry run ruff check $(src) $(other) $(tests)
	poetry run mdformat --check --number .
	poetry run make -C docs clean doctest

format:
	poetry run black $(src) $(other) $(tests)
	poetry run ruff check $(src) $(other) $(tests) --fix
	poetry run mdformat --number .

test:
	poetry run pytest $(tests)

sphinx:
	poetry run make -C docs clean html

open-sphinx:
	open docs/build/html/index.html

install:
	poetry lock && poetry install --all-extras
