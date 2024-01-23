
# Contributed Embedding Ingestion

This package provides support for the processing of Census cell embeddings, both community and CZI generated.

Typical usage follows a simple recipe:

1. Create a working directory _for each embedding_, and in the directory place:
   * the embedding metadata, in a file named `meta.yml`
   * the embedding and joinid values, in NPY or SOMA format
2. `ingest` - given source embeddings in NPY or SOMA formats, along with associated metadata in a YAML file, create a SOMA SparseNDArray. This step will create the SOMA SparseNDArray, and decorate it with associated metadata.
3. `inject` - for embeddings to be published as part of the Census (in an `obsm` layer), this step will add the previously built emdedding to the Census build.

## Stand-alone embeddings

For embeddings to be published stand-alone (i.e., not as part of `obsm`):

1. Create a working directory
2. Create a meta.yml file containing all required fields, populated with contirbutor-provided metadata
3. Run the ingest step.

Embeddings can be provided in one of two formats:

* SOMA SparseNDArray - where dim0 is the obs joinid, dim1 is the user provided feature index, and the embedding is float32
* NPY dense NDArray of embedding, with a second NPY or TXT containing the joinids

Examples, where `working-dir` is the directory containing the metadata and data:

```bash
python -m census_contrib ingest-npy --cwd working-dir --joinid-path joinids.npy --embedding-path embeddings.npy
```

Or alternatively:

```bash
python -m census_contrib ingest-soma --cwd working-dir -v --soma-path a-sparse-soma-array
```

In all cases:

* there must exist a file `meta.yml` in `working-dir` that contains metadata
* the resulting ("ingested") embedding will be written into the `working-dir`, with the accession ID (from the `meta.yml`) as its file name

## Inject embedding into Census

Given a _local copy_ of a Census build, this command will take a previously ingested embedding and add it to the Census build `obsm` collection as a new layer.

Typical usage:

```bash
python -m census_contrib inject --cwd working-dir --census-path path-to-census-build/soma
```

This command:

* places a copy of the embedding into the Census build directory
* adds the embedding array to the `obsm` layer

## Embedding metadata

The [metadata schema definition](embedding_metadata.md) describes the contents of accepted metadata. When stored in a contributed embedding, it is a JSON-encoded string, stored in the SOMANDArray metadata with key `CxG_embedding_info`. The `census_contrib` tool requires this metadata for most operations, and can accept it as either a file named `meta.yml` or `meta.json`.
