# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cell-census'
copyright = '2022-2023 Chan Zuckerberg Initiative'
author = 'Chan Zuckerberg Initiative'

version = "0.5"
release = "0.5.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', "nbsphinx", "sphinx.ext.intersphinx", 'sphinx.ext.napoleon']

tiledb_version = "latest"

intersphinx_mapping = {
    "tiledbsoma-py": (
        "https://tiledb-inc-tiledb-soma.readthedocs-hosted.com/en/%s/"
        % tiledb_version,
        None,
    ),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'anndata': ('https://anndata.readthedocs.io/en/latest/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

def setup(app):
    app.add_css_file("custom.css")