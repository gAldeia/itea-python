from setuptools import find_packages, setup

setup(
    name             = "itea",
    packages         = find_packages(),
    version          = "1.0.20",
    description      = ("Interaction-Transformation Evolutionary Algorithm "
                        "for Symbolic Regression."),
    long_description = open('README.md', encoding='utf-8').read(),
    long_description_content_type = 'text/markdown',
    author           = "Guilherme Aldeia",
    license          = "BSD-3-Clause",
    install_requires = [
        "numpy>=1.18.2",
        "scikit-learn>=0.23.1",
        "matplotlib>=3.2.2",
        "pandas>=1.1.0",
        "jax>=0.2.13",
        "jaxlib>=0.1.67",
        "scipy>=1.5.2",
        "pylatex==1.4.1"
    ],
    package_data = {'examples' : ['examples']},
    include_package_data = True,
    python_requires = '>=3.7',
    setup_requires  = ["wheel", "pytest-runner", "coverage", "coverage-badge"],
    tests_require   = ["pytest", "jax", "statsmodels"],
    test_suite      = "tests",
)