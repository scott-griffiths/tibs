# Configuration file for the Sphinx documentation builder.
#
import os
import time
import datetime

from bitformat._version import VERSION

year = datetime.datetime.utcfromtimestamp(
    int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
).year

project = "bitformat"
copyright = f"2024 - {year}, Scott Griffiths"
author = "Scott Griffiths"
release = VERSION

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.mermaid",
    'enum_tools.autoenum',
    'sphinx_autodoc_typehints',
]
autoapi_dirs = ["../bitformat/"]
autoapi_add_toctree_entry = False
autodoc_mock_imports = ["lark"]

add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "index"

add_function_parentheses = False

html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme = "piccolo_theme"

html_theme_options = {
    # "banner_text": "bitformat is currently in beta. This documentation may be inaccurate.",
    "banner_hiding": "permanent",
    "show_theme_credit": False,
    "globaltoc_collapse": False,
    "source_url": "https://github.com/scott-griffiths/bitformat/",
}

html_logo = "./bitformat_logo_small.png"
html_favicon = "./logo.png"