# Configuration file for the Sphinx documentation builder.
#
import os
import time
import datetime

year = datetime.datetime.utcfromtimestamp(
    int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
).year

project = "tibs"
copyright = f"2025 - {year}, Scott Griffiths"
author = "Scott Griffiths"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    'enum_tools.autoenum',
    'sphinx_autodoc_typehints',
]
autoapi_dirs = ["../tibs/"]
autoapi_add_toctree_entry = False

add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "index"

add_function_parentheses = False

html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_sidebars = {
    "**": []
}

html_theme = "pydata_sphinx_theme"
html_logo = "tibs.png"
html_theme_options = {
    "content_footer_items": ["last-updated"],
    "show_toc_level": 2,
    "logo": {
        # "text": "My awesome documentation",
        "image_light": "tibs.png",
        "image_dark": "tibs.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scott-griffiths/tibs",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/tibs/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "secondary_sidebar_items": ["page-toc"],
}

from pathlib import Path
from bs4 import BeautifulSoup


def process_in_page_toc(app, exception):
    for pagename in app.env.found_docs:
        if not isinstance(pagename, str):
            continue

        with (Path(app.outdir) / f"{pagename}.html").open("r") as f:
            # Parse HTML using BeautifulSoup html parser
            soup = BeautifulSoup(f.read(), "html.parser")

            for li in soup.find_all("li", class_="toc-h3 nav-item toc-entry"):
                if span := li.find("span"):
                    # Modify the toc-nav span element here
                    span.string = span.string.split(".")[-1]

        with (Path(app.outdir) / f"{pagename}.html").open("w") as f:
            # Write back HTML
            f.write(str(soup))


def setup(app):
    app.connect("build-finished", process_in_page_toc)
