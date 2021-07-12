PYTHON = python

EXAMPLES := $(shell find "./examples/"  -maxdepth 1 -name '*.ipynb')

all: coverage doc build-dist clean

profile:
	# before profiling remind of do make all to compile and install the modifications!
	${PYTHON} -m cProfile -o ./profiling/profiling_results.dat ./profiling/itea_profiling.py
	snakeviz ./profiling/profiling_results.dat

test:
	${PYTHON} setup.py pytest

coverage: 
	${PYTHON} -m coverage run --source=. setup.py pytest
	coverage-badge > ./docsource/source/assets/images/coverage.svg
	coverage erase

doc: $(EXAMPLES)
	$(info The following examples will be included in the documentation:)
	$(info [${EXAMPLES}])

	$(foreach example, $(EXAMPLES), $(shell cp $(example) ./docsource/source/$(addprefix _, $(notdir $(example)))))
	
	# May require pip install Jinja2==2.11
	sphinx-build -b html ./docsource/source ./docs
	touch ./docs/.nojekyll

	rm ./docsource/source/_*.ipynb

build-dist: 
	rm -r ./dist/*
	
	${PYTHON} setup.py develop
	${PYTHON} setup.py sdist
	
	${PYTHON} -m pip install ./dist/*.tar.gz

clean:
	rm -r .pytest_cache
	rm `find ./ -name '__pycache__'` -rf

# Upload on test.py:
# python -m twine upload --repository testpypi dist/*