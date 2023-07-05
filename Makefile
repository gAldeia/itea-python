PYTHON = python

EXAMPLES := $(shell find "./examples/"  -maxdepth 1 -name '*.ipynb')

all: build-dist coverage doc clean

profile:
	# before profiling remind of do make all to compile and install the modifications!
	${PYTHON} -m cProfile -o ./profiling/profiling_results.dat ./profiling/itea_profiling.py
	snakeviz ./profiling/profiling_results.dat

test:
	# Create detailed html file. Ignore warnings
	${PYTHON} -m pytest tests/*.py
	
build-dist: 
	rm -f ./dist/*

	${PYTHON} setup.py develop
	${PYTHON} setup.py sdist

	${PYTHON} -m pip install ./dist/*.tar.gz

coverage: 
	${PYTHON} -m coverage run --source=. setup.py pytest
	coverage-badge > ./docsource/source/assets/images/coverage.svg
	coverage erase

copy-notebooks:
	$(info The following examples will be included in the documentation:)
	$(info [${EXAMPLES}])

	# remove previous notebooks, ignore nonexistent
	#rm -f ./docsource/source/_*.ipynb

	$(foreach example, $(EXAMPLES), $(shell cp $(example) ./docsource/source/$(addprefix _, $(notdir $(example)))))

generate-report: ./examples/generate_report_example.py
	${PYTHON} ./examples/generate_report_example.py

	$(shell cp ./examples/Report.pdf ./docsource/source/assets/files/Report.pdf)

doc: $(EXAMPLES) copy-notebooks generate-report
	# May require pip install Jinja2==2.11
	sphinx-build -b html ./docsource/source ./docs
	touch ./docs/.nojekyll

clean:
	rm -r .pytest_cache
	rm `find ./ -name '__pycache__'` -rf

# Upload on test.py: (to upload on pypi remove '--repository testpypi')
# python -m twine upload --repository testpypi dist/*
# user should be __token__, and password is the pypi token (including `pypi-` prefix)