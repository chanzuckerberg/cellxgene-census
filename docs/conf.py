# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cellxgene-census'
copyright = '2022-2023 Chan Zuckerberg Initiative Foundation'
author = 'Chan Zuckerberg Initiative Foundation'

import git
repo = git.Repo(search_parent_directories=True)
tags = [t for t in repo.tags if t.tag is not None]
tags = sorted(tags, key=lambda t: t.tag.tagged_date)
latest_tag = tags[-1]

version = str(latest_tag)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    "nbsphinx", 
    "sphinx.ext.intersphinx", 
    'sphinx.ext.napoleon', 
    'sphinx.ext.autosummary', 
    'myst_parser'
    ]

autosummary_generate = True

napoleon_custom_sections = ["Lifecycle"]

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

source_suffix = ['.rst', '.md']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# Inject custom css files in `/_static/css/*`
html_static_path = ['_static']

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_js_files = [
    ('https://plausible.io/js/script.js', {"data-domain": "chanzuckerberg.github.io/cellxgene-census", "defer": "defer"}),
]

def setup(app):
    app.add_css_file("css/custom.css")
