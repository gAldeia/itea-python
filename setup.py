from setuptools import find_packages, setup
from os import path

root = path.abspath(path.dirname(__file__))
with open(path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name             = "itea",
    packages         = find_packages(include=["itea"]),
    version          = "1.0.0",
    description      = "Interaction-Transformation Evolutionary Algorithm for Symbolic Regression.",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author           = "Guilherme Aldeia",
    license          = "MIT",
    install_requires = ["numpy", "scikit-learn", "matplotlib", "pandas", "jax", "jaxlib", "scipy"],
    setup_requires   = ["pytest-runner", "coverage", "coverage-badge"],
    tests_require    = ["pytest"],
    test_suite       = "tests",
)