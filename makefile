PYTHON = python3

all: test coverage doc build clean

test:
	${PYTHON} -B setup.py pytest

coverage: 
	${PYTHON} -B -m coverage run --source=. setup.py pytest
	coverage-badge > ./docsource/source/assets/images/coverage.svg
	coverage erase

doc:  
	cd docsource
	sphinx-build -b html ./docsource/source ./docs
	touch ./docs/.nojekyll

build: 
	${PYTHON} setup.py bdist_wheel

clean:
	rm -r .pytest_cache
	rm -r itea.egg-info