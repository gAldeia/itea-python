from setuptools import find_packages, setup

description = open('README.md', encoding='utf-8').read()

setup(
    name             = "itea",
    packages         = find_packages(),
    version          = "1.1.1",
    description      = ("Interaction-Transformation Evolutionary Algorithm "
                        "for Symbolic Regression."),
    # Only the "introduction" in README (text below the horizontal rule)
    long_description = description[:description.find('-----')],
    long_description_content_type = 'text/markdown',
    author           = "Guilherme Aldeia",
    author_email="guilherme.aldeia@ufabc.edu.br",
    license          = "BSD-3-Clause",
    install_requires = [
        "numpy>=1.18.2",
        "scikit-learn>=0.23.1",
        "matplotlib>=3.2.2",
        "pandas>=1.1.0",
        "jax>=0.2.13",
        "jaxlib>=0.1.67",
        "scipy>=1.5.2",
        "pylatex==1.4.1",
        "docutils==0.17.1"
    ],
    package_data = {'examples' : ['examples']},
    include_package_data = True,
    python_requires = '>=3.7',
    setup_requires  = [
        "wheel",
        "pytest-runner",
        "coverage",
        "coverage-badge",

        #required to build the docs
        "Jinja2==2.11",  
        "nbsphinx"
    ],

    # Aditional packages not included in install requires
    tests_require   = [
        # Used in tests
        "pytest",
        "pytest-html",

        # Used in profiling
        "snakeviz",

        # Used in benchmarking
        "filelock"
        ],
    test_suite = "tests",
)