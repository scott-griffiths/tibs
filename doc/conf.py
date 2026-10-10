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
release = "2.0"

extensions = [
    "sphinx.ext.autodoc",
    'enum_tools.autoenum',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.mermaid',
    'sphinx_llms_txt',
]

# Read the Docs sets the canonical URL of the version being built. llms.txt links
# are made absolute with it, so they work from /en/latest/ and from release builds.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# llms.txt is an index of the docs for language models, and llms-full.txt is the
# whole manual in one file. Both are built from the .rst sources, which leaves
# autodoc directives unexpanded, so the API reference comes from tibs.pyi instead:
# it is appended to llms-full.txt and served next to it. Its docstrings are
# generated from src/, so nothing here is written twice.
llms_txt_summary = (
    "tibs is a fast Python library for binary data, written in Rust. Tibs is an "
    "immutable sequence of bits and Mutibs a mutable one, with bit positions and "
    "lengths throughout, so fields need not be byte aligned. "
    f"The API reference, with every signature and docstring, is {html_baseurl}tibs.pyi"
)
llms_txt_exclude = ["index"]
llms_txt_code_files = ["+:../tibs.pyi"]

add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "index"

add_function_parentheses = True

html_show_sphinx = False
html_static_path = ["_static"]
html_extra_path = ["tibs.png", "../tibs.pyi"]
html_css_files = ["custom.css"]
html_favicon = 'favicon.png'

html_sidebars = {
    "**": ["sidebar-nav-bs-root.html"],
}

html_theme = "pydata_sphinx_theme"
html_logo = "tibs_static.png"
html_theme_options = {
    "content_footer_items": ["last-updated"],
    "show_toc_level": 2,
    "sidebar_includehidden": True,
    "show_nav_level": 3,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "logo": {
        "text": f"v{release}",
        "image_light": "tibs_static.png",
        "image_dark": "tibs_static.png",
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
    if exception is not None or getattr(app.builder, "format", None) != "html":
        return

    for pagename in app.env.found_docs:
        if not isinstance(pagename, str):
            continue

        html_path = Path(app.outdir) / f"{pagename}.html"
        if not html_path.exists():
            continue

        with html_path.open("r") as f:
            # Parse HTML using BeautifulSoup html parser
            soup = BeautifulSoup(f.read(), "html.parser")

            for li in soup.find_all("li", class_="toc-h3 nav-item toc-entry"):
                if span := li.find("span"):
                    # Modify the toc-nav span element here
                    span.string = span.string.split(".")[-1]

        with html_path.open("w") as f:
            # Write back HTML
            f.write(str(soup))


def setup(app):
    app.connect("build-finished", process_in_page_toc)
