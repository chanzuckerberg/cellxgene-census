# Test README

This directory contains tests of the cell-census package API, _and_ the use of the API on the
live "corpus", i.e., data in the public cell census S3 bucket. The tests use Pytest, and have
Pytest marks to control which tests are run.

Tests can be run in the usual manner. First, ensure you have cell-census installed, e.g., from the top-level repo directory:

> pip install -e ./api/python/cell_census/

Then run the tests:

> pytest ./api/python/cell_census/

## Pytest Marks

There are two Pytest marks you can use from the command line:

- live_corpus: tests that directly access the `latest` version of the Cell Census. Enabled by default.
- expensive: tests that are expensive (ie., cpu, memory, time). Disabled by default - enable with `--expensive`. Some of these tests are _very_ expensive, ie., require a very large memory host to succeed.

To enable `expensive` tests:

> pytest --expensive ...

To disable `live_corpus` tests:

> pytest -m 'not live_corpus'

You can also combine them, e.g.,

> pytest -m 'not live_corpus' --expensive
