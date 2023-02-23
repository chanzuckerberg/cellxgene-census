What is SOMA
=====

The Cell Census is a data object publicly hosted online and a convenience API to open it. 
The object is built using the `SOMA API`_ and data model via its implementation `TileDB-SOMA`_ (`documentation <https://tiledb-inc-tiledb-soma.readthedocs-hosted.com/en/latest/index.html>`_). 

As such, the Cell Census has all the data capabilities offered by TileDB-SOMA and currently absent in the single-cell field, 
including:

- Cloud-based data storage and access.
- Efficient access for larger-than-memory slices of data.
- Data streaming for iterative/parallelizable methods.
- R and Python support.
- Export to AnnData and Seurat.

To get the most value out of the Cell Census it is highly recommended to be familiar with TileDB-SOMA capabilities. 
Please take a look at their `documentation page <https://tiledb-inc-tiledb-soma.readthedocs-hosted.com/en/latest/index.html>`_.

.. _SOMA API: https://github.com/single-cell-data/SOMA
.. _TileDB-SOMA: https://github.com/single-cell-data/TileDB-SOMA