PYTHON = python

all: test coverage doc build-dist clean

test:
	${PYTHON} -B setup.py pytest

coverage: 
	${PYTHON} -B -m coverage run --source=. setup.py pytest
	coverage-badge > ./docsource/source/assets/images/coverage.svg
	coverage erase

doc:
	cp ./examples/regression_example.ipynb ./docsource/source
	cp ./examples/multiclass_example.ipynb ./docsource/source
	cp ./examples/agnostic_explainers.ipynb ./docsource/source

	sphinx-build -b html ./docsource/source ./docs
	touch ./docs/.nojekyll

	rm ./docsource/source/regression_example.ipynb
	rm ./docsource/source/multiclass_example.ipynb
	rm ./docsource/source/agnostic_explainers.ipynb

build-dist: 
	rm -r ./dist/*.whl
	rm -r ./dist/*.egg
	
	${PYTHON} -B setup.py bdist_wheel
	${PYTHON} -B setup.py install
	${PYTHON} -B -m pip install ./dist/*.whl

clean:
	rm -r .pytest_cache
	rm `find ./ -name '__pycache__'` -rf

# python3 -m twine upload --repository testpypi dist/*