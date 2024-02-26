# vignettes_/ and pkgdown docs

Our vignettes use too much time and memory to build in the routine `R CMD build` procedure (especially in limited-memory GitHub Actions workers used by r-universe, which also doesn't allow build flags like `--no-build-vignettes`). For this reason, the vignettes are stored here under [`vignettes_/`](vignettes_) instead of `vignettes/`, to hide them from `R CMD build`.

The vignettes *are* included in our pkgdown documentation site. To build them, the [`update_docs.sh`](../update_docs.sh) script locally+temporarily symlinks `vignettes/` to `vignettes_/` so that pkgdown finds them. We then check in the resulting `docs/` folder to git, which the [docsite-build-deploy workflow](../../../../.github/workflows/docsite-build-deploy.yml) later copies into the docsite.

1. Ensure R packages up-to-date: tiledbsoma rmarkdown pkgdown
2. Run `api/r/cellxgene.census/update_docs.sh`
3. Stage the updated pkgdown site into git: `git rm -r --cached api/r/cellxgene.census/docs/ && git add api/r/cellxgene.census/docs/`
4. Review and commit
