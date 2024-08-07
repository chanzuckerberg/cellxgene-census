# Census Geneformer training and embeddings generation

These scripts automate:

1. preparing tokenized Geneformer datasets from CELLxGENE Census (`prepare-census-geneformer-dataset.py`)
2. fine-tuning a Geneformer cell classifier model (`finetune-geneformer.py`)
3. using the fine-tuned model to generate cell embedding vectors (`generate-geneformer-embeddings.py`)

Embedding generation is computationally intensive on large datasets (e.g. all of Census). To make this practical, a WDL workflow (`wdl/generate_embeddings.wdl`) provides a way to distribute across many compute nodes. The other steps also have WDLs for encapsulation, even though they aren't distributed.

The `Dockerfile` provides the recipe for the docker image used by the WDLs, which packages the scripts together with `cellxgene_census`, Geneformer, pytorch, etc. It also bundles `finetune-geneformer.config.yml` with various fine-tuning settings; an alternate config file can be supplied at runtime.

## Example invocations

Using [miniwdl-omics-run](https://github.com/miniwdl-ext/miniwdl-omics-run) for the Amazon HealthOmics workflow service, and assuming the docker image has been built and pushed to a suitable repository like ECR (tagged `$DOCKER_TAG`).

Preparing a tokenized training dataset with 2,500 primary cells per human cell type:

```bash
miniwdl-omics-run wdl/prepare_datasets.wdl \
    docker=$DOCKER_TAG \
    census_version=s3://cellxgene-census-public-us-west-2/cell-census/2023-12-15/soma/ \
    N=2500 sampling_column=cell_type output_name=2500_per_cell_type \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/datasets/
```

And a tokenized dataset for all of Census (>300GiB, sharded):

```bash
miniwdl-omics-run wdl/prepare_datasets.wdl \
    docker=$DOCKER_TAG \
    census_version=s3://cellxgene-census-public-us-west-2/cell-census/2024-05-20/soma/ \
    value_filter='is_primary_data==True or is_primary_data==False' \
    output_name=2024-05-20 shards=256 \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/datasets/
```

(We set `census_version` to the SOMACollection URI because the HealthOmics workers don't have internet access to the Census release directory endpoint.)

Fine-tuning for 8 epochs (takes ~36h on g5.8xlarge):

```bash
miniwdl-omics-run wdl/finetune_geneformer.wdl \
    docker=$DOCKER_TAG \
    dataset=s3://MYBUCKET/geneformer/datasets/2500_per_cell_type/dataset/2500_per_cell_type \
    epochs=8 output_name=2500_per_cell_type_8epochs \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/models/
```

Generating cell embeddings (takes 8-12h on up to 256 g5.2xlarge, generates 130GiB `tiledbsoma.SparseNDArray` on S3):

```bash
seq 0 255 \
    | xargs -n 1 printf 'dataset_shards=s3://MYBUCKET/geneformer/datasets/census-2024-05-20/shard-%03d/\n' \
    | xargs -n 9999 miniwdl-omics-run \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/embs \
    wdl/generate_embeddings.wdl \
    docker=$DOCKER_TAG \
    emb_layer=0 model_type=Pretrained \
    model=s3://MYBUCKET/geneformer/gf-95m/fine_tuned_model/ \
    output_uri=s3_//MYBUCKET/geneformer/embs/$(date '+%s')/census-2024-05-20/
```

(The `s3_//MYBUCKET` is a workaround for the workflow service rejecting our submission if the specified S3 output folder doesn't yet exist; this workflow has TileDB create it.)
