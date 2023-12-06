# Differential Expression using memento

This directory contains code for a Census-integrated version of the `memento` method for differential expression
analysis, including differential variability and co-expression. The underlying method is described in
the [memento pre-print](https://www.biorxiv.org/content/10.1101/2022.11.09.515836v1).

This implementation relies upon a database of pre-computed estimators that are derived from a given Census data release.
The database is a TileDB array, structured as a multi-dimensional cube. It is built by
the `tools/models/memento/src/estimators_cube_builder/cube_builder.py` script.
