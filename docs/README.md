# API Documentation

The documentation website is currently hosted on <https://chanzuckerberg.github.io/cellxgene-census/>.

The documentation site is rebuilt each time a tag is created on the repo, which happens on release, including regenerating the Sphinx Python API docs. The R `pkgdown` website is checked into git and simply copied in during the doc site rebuild; see [`api/r/cellxgene.census/vignettes_/`](https://github.com/chanzuckerberg/cellxgene-census/tree/main/api/r/cellxgene.census/vignettes_) for further explanation.

A docsite rebuild can be [triggered manually through `workflow_dispatch`](https://github.com/chanzuckerberg/cellxgene-census/actions/workflows/docsite-build-deploy.yml) (Run workflow). This should be done if a bug in the documentation is found and a release is not necessary.

In order to test docsite changes locally, first install the necessary requirements:

```shell
pip install -r docs/requirements.txt
brew install pandoc # Mac OS
```

Then,

And then run the following command:

```shell
cd docs
make html
```

The generated docsite will then be found at `docs/_build/html/index.html`.
