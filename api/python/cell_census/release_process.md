# cell_census Python package release process

The following approach is used to manage releases of the Python cell_census package:

1. The cell_census package is automatically built (sdist and wheels) in a GitHub action, and build artifacts are uploaded to GitHub.
2. Release candidate testing is done by installing built assets from Github.
3. Build versions are managed via [`setuptools_scm`](https://github.com/pypa/setuptools_scm) and the version is automatically determined from git tags.
4. Releases are created and managed via GitHub Releases, leaving a tag in place from which future branches (eg, emergency fixes) can be created.
5. Built packages are published to PyPi _from_ the GitHub assets, i.e. are never hand-created, to reduce errors.

## Prerequisites

While not strictly required, this process assumes you have met the following prerequisites:

- Write access to the chanzuckerberg/cell_census repo
- An account on pypi.org and test.pypi.org with access to the cell_census project
- Github CLI tool installed. See [documentation](https://cli.github.com/).
- `pipx` CLI tool installed. See [documentation](https://pypa.github.io/pipx/).

## Step 1: Building the package assets

A build will occur automatically upon each commit to main, or upon each commit to a PR. Builds are retained for a limited period of time (currently the GH default of 90 days).

## Step 2: Release candidate testing

Any pre-built asset on Github can be installed and tested from the Github URL. For example:

- Download the build artifact.zip from GitHub, using the GH Action run ID associated with the `build` action for your commit:
  ```shell
  $ gh run download <id>
  ```
- If you downloaded using the browser, unzip into a temp directory, e.g.,
  ```shell
  $ unzip artifact.zip -d ./artifact/
  Archive:  artifact.zip
    inflating: ./artifact/cell_census-0.0.1.dev0-py3-none-any.whl
    inflating: ./artifact/cell_census-0.0.1.dev0.tar.gz
  ```
- Uninstall any existing installation, install and test the downloaded version, e.g.,
  ```shell
  $ pip uninstall cell_census
  $ pip install ./artifact/cell_census-0.0.1.dev0-py3-none-any.whl
  ```

To test a release candidate:

1. Identify the commit tag you wish to test. Download and test the artifact built for that commit as described above.
2. Perform any necessary end-user testing, using the above installation method
3. If acceptable, proceed to Step 3 - create a release.

If testing exposes problems, fix and commit a solution as you would any other change.

## Step 3: Create a release

Prior to this process, determine the correct semver version number for the new release. Please consider if this is a major, minor or patch release, and if it should have a tag (e.g., a release candidate, with a `-rc.#` suffix).

This process also assumes you are releasing from `main`. If you create a release PR, it should be merged before releasing.

To create a release, perform the following:

1. Identify both the (tested & validated) commit and semver for the release.
2. Tag the commit with the release version (_including_ a `v` prefix) and push the tag to origin. **Important**: use an annotated tag, e.g., `git tag -a v1.9.4 -m 'Release 1.9.4`. For example:
   ```shell
   $ git tag -a v1.9.4 -m 'Release 1.9.4'
   $ git push origin v1.9.4
   ```
3. Trigger a build by manually triggering the `build.yml` workflow. For example:
   ```shell
   $ gh workflow run build.yml --ref v1.9.4
   ```
4. When the workflow completes, make note of the run ID (e.g., using `gh run list`).
5. Optional: download the asset from the build workflow and validate it.
6. Create and publish a GitHub Release, using the tag above (e.g., `v1.9.4`).).

## Step 4: Publish assets to PyPi

To publish built release assets to PyPi (_note_: this will require your pypi login):

1. Download the assets built for your release commit, using the same method as step 2 above, e.g.,
   ```shell
   $ gh run download <ID>
   ```
2. Use twine to upload to PyPi (this assumes the downloaded assets are in ./artifact/), e.g.,
   ```shell
   pipx run twine upload ./artifact/*
   ```

To test with TestPyPi, use `twine upload --repository testpypi artifact/*`. More instructions [here](https://packaging.python.org/en/latest/guides/using-testpypi/).
