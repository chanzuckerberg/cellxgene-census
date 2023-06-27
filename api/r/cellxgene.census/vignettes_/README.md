# vignettes_/

The `cellxgene.census` R package vignettes reside in this directory instead of `vignettes/` because they use too much time & memory to evaluate in the routine R package build process (including [the r-universe build](https://chanzuckerberg.r-universe.dev/builds), which runs on GitHub Actions workers with only 7G RAM and doesn't allow us to set `--no-vignettes`).

The [build_vignettes.sh](build_vignettes.sh) script builds each Rmarkdown vignette into a static HTML file alongside. Run this on a suitable machine and check the updated HTML files into git.

The [docsite-build-deploy workflow](../../../../.github/workflows/docsite-build-deploy.yml) copies the HTML files into the docsite on GitHub Pages, and [cellxgene_census_docsite_r_tutorials.md](../../../../docs/cellxgene_census_docsite_r_tutorials.md) links to them from the docsite TOC. Note that markdown file should be updated manually if a vignette is added or renamed.
