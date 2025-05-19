# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RVV-SIMD'
copyright = '2025, RISC-V Vector Library Team'
author = 'RISC-V Vector Library Team'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    # 'breathe',
    # 'exhale',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'pt_BR'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'logo_only': False,
}

# -- Breathe configuration -------------------------------------------------
breathe_projects = {
    "RVV-SIMD": "./doxygen/xml"
}
breathe_default_project = "RVV-SIMD"
breathe_default_members = ('members', 'undoc-members')

# -- Exhale configuration -------------------------------------------------
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "API da Biblioteca",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    """
        INPUT                  = ../include ../src
        RECURSIVE              = YES
        EXCLUDE_PATTERNS       = */tests/* */examples/*
        EXTRACT_ALL            = YES
        EXTRACT_PRIVATE        = YES
        EXTRACT_STATIC         = YES
        GENERATE_HTML          = NO
        GENERATE_XML           = YES
        XML_OUTPUT             = xml
    """
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
