# cellxgene_census Python package release process

The following approach is used to manage releases of the Python cellxgene_census package:

1. The cellxgene_census package is automatically built (sdist and wheels) in a GitHub action, and build artifacts are uploaded to GitHub.
2. Release candidate testing is done by installing built assets from Github.
3. Build versions are managed via [`setuptools_scm`](https://github.com/pypa/setuptools_scm) and the version is automatically determined from git tags.
4. Releases are created and managed via GitHub Releases, leaving a tag in place from which future branches (eg, emergency fixes) can be created.
5. Built packages are published to PyPi _from_ the GitHub assets, i.e. are never hand-created, to reduce errors.

## Prerequisites

While not strictly required, this process assumes you have met the following prerequisites:

- You have write access to the `chanzuckerberg/cellxgene-census` repo
- You have an account on pypi.org and test.pypi.org, both with access to the cellxgene_census project. You will need to have created an API token on each account so that you can authenticate to test.pypi.org and pypi.org accounts when using `twine`. Usually this means adding these tokens to your `~/.pypirc` file. See https://pypi.org/help/#apitoken for more information.
- You have the Github CLI tool (`gh`) installed. See [documentation](https://cli.github.com/).
- You have the `pipx` CLI tool installed. See [documentation](https://pypa.github.io/pipx/). 

## Step 1: Building the package assets

A build will occur automatically upon each commit to main, upon each commit to a PR, or when the build workflow is manually invoked. The build workflow is defined by the GH `py-build.yml` workflow. Build artifacts are the Python setuptools-created `sdist` and `wheel`, and are retained for a limited period of time (currently the GH default of 90 days).

Unless you are revising and testing the build process itself, there is no need to manually perform a build.

## Step 2: Release candidate testing

Any pre-built asset on Github can be installed and tested from the Github URL. For example:

1. Identify the GH workflow run ID that contains the asset you wish to test. A simple way to do this is:
   ```shell
   $ gh run list
   ```
   Alternatively, you can use the "Actions" tag in the GitHub web UI.
2. Download the build artifact.zip from GitHub, using the GH Action run ID associated with the `build` action for your commit OR utilizing the web UI:

   ```shell
   $ gh run download <ID>
   ```

   If you download using the browser, unzip into a temp directory, e.g.,

   ```shell
   $ unzip artifact.zip -d ./artifact/
   Archive:  artifact.zip
     inflating: ./artifact/cellxgene_census-0.0.1.dev0-py3-none-any.whl
     inflating: ./artifact/cellxgene_census-0.0.1.dev0.tar.gz
   ```

3. Install and test the downloaded build, e.g.,
   ```shell
   $ pip uninstall cellxgene-census
   $ pip install ./artifact/cellxgene_census-*-any.whl
   ```

To test a release candidate:

1. Identify the build you wish to test. Download and test the artifact built for that commit as described above.
2. Perform end-user testing, using the above installation method
3. If acceptable, proceed to Step 3 - create a release.

If testing exposes problems, fix and commit a solution as you would any other change.

## Step 3: Create a release

Prior to this process, determine the correct semver version number for the new release. Please consider if this is a major, minor or patch release, and if it should have a tag (e.g., an alpha build, with a `a#` suffix or a pre-release candidate, with a `-rc` suffix). If you are not sure, please review [PEP 440](https://peps.python.org/pep-0440/) for more information.

This process also assumes you are releasing from `main`. If you create a release PR, it should be merged to main before releasing.

To create a release, perform the following:

1. Identify both the (tested & validated) commit and semver for the release.
2. Tag the commit with the release version (_including_ a `v` prefix) and push the tag to origin. **Important**: use an annotated tag, e.g., `git tag -a v1.9.4 -m 'Release 1.9.4`. For example (please replace <SEMVER> with your version, _including_ a `v`, e.g. `v1.9.4`:
   ```shell
   $ git tag -a <SEMVER> -m 'Release <SEMVER>'
   $ git push origin <SEMVER>
   ```
3. Trigger a build for this tag by manually triggering the `py-build.yml` workflow. For example:
   ```shell
   $ gh workflow run py-build.yml --ref <SEMVER>
   ```
4. When the workflow completes, make note of the run ID (e.g., using `gh run list`).
5. Optional, _but recommended_: download the asset from the build workflow and validate it.
6. Create and publish a GitHub Release [here](https://github.com/chanzuckerberg/cellxgene-census/releases/new). Set the release title to the `<SEMVER>`. Select `Set as the latest release`. Use the `Generate Release Notes` button to auto-populate the summary with a changelog. It is reasonable to remove any R-specific or builder-specific entries. Add a prelude to the summary, noting any major new features or API changes. 

## Step 4: Publish assets to PyPi

To publish built release assets to PyPi (_note_: this will require your pypi/testpypi login):

1. Delete any existing release builds you may have accumulated in the past: `rm ./artifact/*`.
2. Download the assets built for your release commit, using the same method as step 2 above, e.g.,
   ```shell
   $ gh run download <ID>
   ```
3. Optional: upload to TestPyPi (this assumes the downloaded assets are in ./artifact/).

   ```shell
   pipx run twine upload --repository testpypi ./artifact/*
   ```

   Following the upload, confirm correct presentation on the project page and ability to download install from TestPyPi. 
4. To test installation from TestPyPi:
   ```shell
   pip install --no-cache-dir -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cellxgene-census
   python -c "import cellxgene_census; print(cellxgene_census.__version__)"
   ```
   Note that the `--extra-index-url` option ensures that any transitive package dependencies that are _not_ available on
   `test.pypi.org` can be satisfied by installing them from the production `pypi.org`.  You can find more information
   [here](https://packaging.python.org/en/latest/guides/using-testpypi/).
5. Use twine to upload to PyPi (this assumes the downloaded assets are in ./artifact/), e.g.,
   ```shell
   pipx run twine upload ./artifact/*
6. Test the installation from PyPi, as a final sanity check. Note that it may take a minute for the new release to be visible on pypi.org:
   ```shell
   pip install --no-cache-dir cellxgene-census
   python -c "import cellxgene_census; print(cellxgene_census.__version__)"
   ```
