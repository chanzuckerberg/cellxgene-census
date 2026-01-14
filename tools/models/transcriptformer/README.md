# Census TranscriptFormer embeddings generation

This directory has a workflow to generate [TranscriptFormer](https://github.com/czi-ai/transcriptformer) embeddings for Census. It scales up the `transcriptformer inference` subcommand built-in to the [PyPI package](https://pypi.org/project/transcriptformer/) by sharding Census and parallelizing on a fleet of GPU workers.

Workflow outline:

1. Use the [Census API](https://chanzuckerberg.github.io/cellxgene-census/python-api.html) to list all desired obs IDs (cell IDs). ([planner.py](planner.py))
2. Shard the list for distribution to a fleet of g6e.xlarge inference workers.
    - The parallel workflow is implemented in [WDL](https://github.com/openwdl/wdl) meant to run on [HealthOmics](https://aws.amazon.com/healthomics/). ([census_transcriptformer.wdl](census_transcriptformer.wdl))
    - But the workflow structure is simple enough to express in any reasonable Docker orchestrator.
3. On each worker node, further subdivide its obs ID shard into "megabatches" (as distinct from torch batches). ([inference_driver.py](inference_driver.py))
    - Megabatch size is a 'Goldilocks' value:
        - Too low: won't amortize startup cost of `transcriptformer inference` loading/instantiating the model
        - Too high: `transcriptformer inference` runs out of main memory preparing input tensors
4. For each megabatch of obs IDs, [get AnnData](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.get_anndata.html#cellxgene_census.get_anndata) with the cells' RNA counts and save h5ad.
5. Run each h5ad through `transcriptformer inference`, generating the 2048-dimensional embedding vector for each cell.
6. Deposit the embedding vectors to a `tiledbsoma.SparseNDArray` on S3 ([put_embeddings.py](put_embeddings.py))
    - TileDB on S3 is a convenient "sink" because it allows all the workers to write into one logical array, without a separate "gather" step.
    - The array is meant to be postprocessed by [census_contrib](https://github.com/chanzuckerberg/cellxgene-census/tree/main/tools/census_contrib) for eventual publication. We go a little out of our way to write the array using the same version of `tiledbsoma` that `census_contrib` uses (typically older than the version used by `cellxgene_census`), ensuring compatibility.
    - As an optimization, while `transcriptformer inference` is running on one megabatch, in the background we're writing the embeddings from the prior megabatch and preparing the next megabatch h5ad.

The [Dockerfile](Dockerfile) bundles these scripts along with `transcriptformer` and all dependencies. Build and push it to ECR using a dev instance or CodeBuild, since it's >10GiB and thus rather painful to push over office/home WiFi.

Example launch invocation using [miniwdl-omics-run](https://github.com/miniwdl-ext/miniwdl-omics-run):

```bash
miniwdl-omics-run \
    --role poweromics --output-uri s3://${WORK_BUCKET}/transcriptformer/embs \
    census_transcriptformer.wdl \
    docker=${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/omics:census-transcriptformer \
    output_uri=s3_//${WORK_BUCKET}/transcriptformer/embs/$(date '+%s')/ \
    organism='Mus musculus' \
    batch_size=48 \
    megabatch_size=16384 \
    shards=96 \
    model=tf_exemplar --name mouse-tf-exemplar
```

(The `s3_//` instead of `s3://` for `output_uri` isn't a typo, but a workaround for the workflow service rejecting our submission if that S3 folder doesn't yet exist; the workflow creates it using TileDB.)
