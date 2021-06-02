PYTHON = python3

all: test coverage doc build clean

test:
	${PYTHON} -B setup.py pytest

coverage: 
	${PYTHON} -B -m coverage run --source=. setup.py pytest
	coverage-badge > ./docsource/source/assets/images/coverage.svg
	coverage erase

doc:
	cp ./examples/regression_example.ipynb ./docsource/source
	cp ./examples/multiclass_example.ipynb ./docsource/source

	sphinx-build -b html ./docsource/source ./docs
	touch ./docs/.nojekyll

	rm ./docsource/source/regression_example.ipynb
	rm ./docsource/source/multiclass_example.ipynb

build: 
	rm -r ./dist/*
	${PYTHON} -B setup.py bdist_wheel
	${PYTHON} -B -m pip install ./dist/*

clean:
	rm -r .pytest_cache
	rm -r itea.egg-info

# python3 -m twine upload --repository testpypi dist/*