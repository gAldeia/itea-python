from setuptools import find_packages, setup

setup(
    name             = "itea",
    packages         = find_packages(include=["itea"]),
    version          = "0.1.0",
    description      = "Interaction-Transformation Evolutionary Algorithm for Symbolic Regression.",
    author           = "Guilherme Aldeia",
    license          = "MIT",
    install_requires = ["numpy", "scikit-learn", "matplotlib", "pandas", "jax", 'scipy'],
    setup_requires   = ["pytest-runner"],
    tests_require    = ["pytest"],
    test_suite       = "tests",
)