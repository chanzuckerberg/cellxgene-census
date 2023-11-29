# Community Contributed Embedding Schema

Each embedding is encoded as a SOMA SparseNDArray, where:

* dimension 0 (`soma_dim_0`) encodes the cell (obs) `soma_joinid` value
* dimension 1 (`soma_dim_1`) encodes the embedding feature, and is in the range [0, N) where N is the number of featues in the embedding
* data (`soma_data`) is float32

⚠️ **IMPORTANT:** Community-contributed embeddings may embed a subset of the cells in any given Census version. If a cell has an embedding, it will be explicitly stored in the sparse array, _even if the embedding value is zero_. In other words, missing array values values imply that the cell was not embedded, whereas zero valued embeddings are explicitly stored. Put another way, the `nnz` of the embedding array indicate the number of embedded cells, not the number of non-zero values.

The first axis of the embedding array will have the same shape as the corresponding `obs` DataFrame for the Census build and experiment. The second axis of the embedding will have a shape (0, N) where N is the number of features in the embedding.

Embedding values are stored as a float32, and are precision reduced to the precision of a [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), i.e., have 8 bits of exponent and 7 bits of mantissa.

## Metadata

Each embedding will contain a variety of metadata stored in the SOMA `metadata` slot, encoded as a JSON string. This metadata includes the following fields encoded as a dictionary:

| Field name             | Optionality | Type    | Description                                                                                      |
| ---------------------- | ----------- | ------- | ------------------------------------------------------------------------------------------------ |
| id                     | required    | string  | CZI-assigned accession ID for this embedding                                                     |
| title                  | required    | string  | Brief project title                                                                              |
| description            | required    | string  | Succinct description of the method and characteristics of the embeddings and model               |
| contact_name           | required    | string  | Name of corresponding author.                                                                    |
| contact_email          | required    | string  | Email address for the corresponding author.                                                      |
| contact_affiliation    | required    | string  | Corresponding athor's affiliation, eg., Brown Lab, Chicago University                            |
| DOI                    | optional    | string  | DOI or URL                                                                                       |
| additional_information | optional    | string  | Additional information on method or embedding.                                                   |
| model_link             | optional    | string  | URL - link to models hosted for public access, e.g., Hugging Face, Google Drive, etc.            |
| data_type              | required    | string  | Data type. Currently one of "obs_embedding" (cell embedding) or "var_embedding" (gene embedding) |
| census_version         | required    | string  | The Census version associated with the embeddings, expressed as a Census tag (e.g., 2023-10-23)  |
| experiment_name        | required    | string  | Experiment (organism), currently one of "homo_sapiens" or "mus_musculus"                         |
| measurement_name       | required    | string  | The measurement name embeddings are associated with, normally "RNA"                              |
| n_embeddings           | required    | integer | Number of embedded cells, integer.                                                               |
| n_features             | required    | integer | Number of features (embedding vector size per cell), integer.                                    |
| submission_date        | required    | string  | Date the contribution was received, ISO 8601 date, YYYY-MM-DD                                    |
