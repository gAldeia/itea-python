from setuptools import find_packages, setup

description = open('README.md', encoding='utf-8').read()

setup(
    name             = "itea",
    packages         = find_packages(),
    version          = "1.1.2b0",
    description      = ("Interaction-Transformation Evolutionary Algorithm "
                        "for Symbolic Regression."),
    # Only the "introduction" in README (text below the horizontal rule)
    long_description = description[:description.find('-----')],
    long_description_content_type = 'text/markdown',
    author           = "Guilherme Aldeia",
    author_email     = "guilherme.aldeia@ufabc.edu.br",
    license          = "BSD-3-Clause",
    install_requires = [
        # ITEA core
        "numpy",        # >=1.18.2
        "scikit-learn", # >=1.0.0
        "jax[cpu]",
        "scipy",        # >=1.5.2

        # ITEA summarizer
        "matplotlib",   # >=3.2.2
        "pandas",       # >=1.1.0
        "pylatex"       # >=1.4.1
    ],
    package_data = {'examples' : ['examples']},
    include_package_data = True,
    python_requires = '>=3.7',
    setup_requires  = [
        "setuptools",
        "wheel"
    ],

    # Aditional packages not included in install requires
    tests_require   = [
        "pytest-runner",
        "pytest"
    ],
    test_suite = "tests",
)