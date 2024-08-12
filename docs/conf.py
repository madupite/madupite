# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from enum import auto
import inspect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "madupite"
copyright = "2024, Philip Pawlowsky, Robin Sieber"
author = "Philip Pawlowsky, Robin Sieber"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]


def setup(app):
    from sphinx.util import inspect

    wrapped_isfunc = inspect.isfunction

    def isfunc(obj):
        type_name = str(type(obj))
        if "nanobind.nb_method" in type_name or "nanobind.nb_func" in type_name:
            return True
        return wrapped_isfunc(obj)

    inspect.isfunction = isfunc

autosummary_generate = True

html_theme = "pydata_sphinx_theme"

html_title = "madupite"

html_logo = "_static/madupite_logo.png"

html_theme_options = {
    "footer_end": ["theme-version", "last-updated"],
#    "secondary_sidebar_items" : ["edit-this-page"],
    "header_links_before_dropdown": 10,
    "navigation_with_keys":True,
    'nosidebar': True,
    "logo": {
        "text": "madupite",
        "image_dark": "_static/madupite_logo.png",
    },
}

html_sidebars = {
  "**": []
}

exclude_patterns = []

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = ".rst"
master_doc = "index"
