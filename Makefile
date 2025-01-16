src := reverse_feature_selection
other := feature_selection_benchmark
test := test

check:
	poetry run black $(src) $(other) $(test) --check --diff
	poetry run mypy --install-types --non-interactive --check-untyped-defs $(src) $(other) $(test)
	poetry run ruff check $(src) $(other) $(test)
	poetry run mdformat --check --number .
	poetry run make -C docs clean doctest

format:
	poetry run black $(src) $(other) $(test)
	poetry run ruff check $(src) $(other) $(test) --fix
	poetry run mdformat --number .

test:
	poetry run pytest $(test)

sphinx:
	poetry run make -C docs clean html

open-sphinx:
	open docs/build/html/index.html

install:
	poetry lock && poetry install --all-extras
