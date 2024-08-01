# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "madupite"
copyright = "2024, Philip Pawlowsky, Robin Sieber"
author = "Philip Pawlowsky, Robin Sieber"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
]

highlight_language = "cython"

exclude_patterns = []

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = ".rst"
master_doc = "index"
