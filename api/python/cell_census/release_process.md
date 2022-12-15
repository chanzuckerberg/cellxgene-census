# cell_census Python package release process

The following approach is used to manage releases of the Python cell_census package:

1. The cell_census package is automatically built (sdist and wheels) in a GitHub action, and build artifacts are uploaded to GitHub.
2. Release candidate testing is done by installing built assets from Github.
3. Releases are created and managed via GitHub Releases, leaving a tag in place from which future branches (eg, emergency fixes) can be created.
4. Built packages are published to PyPi _from_ the GitHub assets, i.e. are never hand-created, to reduce errors.

## Step 1: Building the package assets

A build will occur automatically upon each commit to main, or upon each commit to a PR.

## Step 2: Release candidate testing

Any pre-built asset on Github can be installed and tested from the Github URL. For example:

- Download the build artifact.zip from GitHub
- Unzip into a temp directory, e.g.,
  ```shell
  $ unzip artifact.zip -d /tmp/dist/
  Archive:  artifact.zip
    inflating: /tmp/dist/cell_census-0.0.1.dev0-py3-none-any.whl
    inflating: /tmp/dist/cell_census-0.0.1.dev0.tar.gz
  ```
- Install and test, e.g.,
  ```shell
  $ pip install /tmp/dist/cell_census-0.0.1.dev0-py3-none-any.whl
  ```

To test a release candidate:

1. Identify the commit tag you wish to test. Download and test the artifact built for that commit as desribed above.
2. Perform any necessary end-user testing, using the above installation method
3. If acceptable, proceed to Step 3 - create a release.

If testing exposes problems, fix and commit a solution as you would any other change.

## Step 3: Create a release

Prior to this process, determine the correct semver version number for the new release. Please consider if this is a major, minor or patch release, and if it should have a tag (e.g., a release candidate, with a `-rc.#` suffix).

To create a release, perform the following:

1. Create a branch and PR at the commit/tag you wish to release the repo. Title the PR so that it is obvious that it is a release PR (e.g., "Release 1.0.3", "Release Candidate 2.9.0-rc.1", etc.).
2. Using the new semver, set the version in `api/python/cell_census/src/cell_census/__init__.py` and commit/push this change upstream.
3. Wait for the GitHub action to build the assets, and ensure the CI passes.
4. Download the asset zip, and unzip it into a temporary directory (e.g., /tmp/dist/).
5. If desired, test the assets using instructions from Step 2 above.
6. If this is a final release (not a release candiate), merge the PR into main. If a relase candidate, with the final release expected shortly, maintain the PR and repeat the above steps for the final release.
7. Create and publish a GitHub Release, using the semver release number defined above (prefixed with `v`) as a tag. If this is a release candiate, mark as a pre-release in the GitHub Release form.

Do not delete the build assets - proceed to publishing assets to PyPi in Step 4

## Step 4: Publish assets to PyPi

This step assumes you have a PyPi account with permission to manage the project. You may optionally test this process with test.pypi.org.

To publish built release assets to PyPi:

1. Locate the assets you downloaded in Step 3
2. Use twine to upload to PyPi:
   ```shell
   pipx run twine upload /tmp/dist/*
   ```
