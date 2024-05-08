src := src

# check the code
check:
	pydocstyle --count $(src)
	black $(src) --check --diff
	flake8 $(src)
	isort $(src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src)
	pylint $(src) $(other-src)

# format the code
format:
	black $(src) 
	isort $(src)
	mdformat *.md

install:
	poetry lock && poetry install --all-extras
