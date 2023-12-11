# Census Geneformer training and embeddings generation

These scripts automate:

1. preparing tokenized Geneformer datasets from CELLxGENE Census (`prepare-census-geneformer-dataset.py`)
2. fine-tuning a Geneformer cell classifier model (`finetune-geneformer.py`)
3. using the fine-tuned model to generate cell embedding vectors (`generate-geneformer-embeddings.py`)

Embedding generation is computationally intensive on large datasets (e.g. all of Census). To make this practical, a WDL workflow (`wdl/generate_embeddings.wdl`) provides a way to distribute across many compute nodes. The other steps also have WDLs for encapsulation, even though they aren't distributed.

The `Dockerfile` provides the recipe for the docker image used by the WDLs, which packages the scripts together with `cellxgene_census`, Geneformer, pytorch, etc. It also bundles `finetune-geneformer.config.yml` with various fine-tuning settings; an alternate config file can be supplied at runtime.

## Example invocations

Using a [miniwdl-aws](https://github.com/miniwdl-ext/miniwdl-aws) deployment with suitable GPU instance types enabled on the underlying AWS Batch compute environment, and assuming the docker image has been built and pushed to a suitable repository like ECR (tagged `$DOCKER_TAG`).

Preparing a tokenized training dataset with 2,500 primary cells per human cell type:

```bash
miniwdl-aws-submit --verbose --follow --workflow-queue miniwdl-workflow \
    wdl/prepare_datasets.wdl docker=$DOCKER_TAG \
    census_version=2023-10-23 N=2500 sampling_column=cell_type output_name=2500_per_cell_type \
    --s3upload s3://MYBUCKET/geneformer/datasets/2500_per_cell_type/
```

And a tokenized dataset for all of Census (371GiB!):

```bash
miniwdl-aws-submit --verbose --follow --workflow-queue miniwdl-workflow \
    wdl/prepare_datasets.wdl docker=$DOCKER_TAG \
    census_version=2023-10-23 output_name=census-2023-10-23 value_filter='is_primary_data==True or is_primary_data==False' \
    --s3upload s3://MYBUCKET/geneformer/datasets/census-2023-10-23/
```

Fine-tuning for 8 epochs (takes ~36h on g5.8xlarge):

```bash
MINIWDL__AWS__GPU_VALUE=8 \
MINIWDL__AWS__CONTAINER_PROPERTIES='{"linuxParameters":{"sharedMemorySize":4096}}' \
miniwdl-aws-submit --verbose --follow --workflow-queue miniwdl-workflow \
    wdl/finetune_geneformer.wdl docker=$DOCKER_TAG \
    dataset=s3://MYBUCKET/geneformer/datasets/2500_per_cell_type/dataset/2500_per_cell_type \
    epochs=8 output_name=2500_per_cell_type_8epochs \
    --s3upload s3://MYBUCKET/geneformer/models/2500_per_cell_type_8epochs/
```

Generating cell embeddings (takes 8-12h on up to 256 g5.2xlarge, generates 130GiB `tiledbsoma.SparseNDArray` on S3):

```bash
MINIWDL__SCHEDULER__CALL_CONCURRENCY=256 \
MINIWDL__AWS__SUBMIT_PERIOD=60 \
miniwdl-aws-submit --verbose --follow --workflow-queue miniwdl-workflow \
    wdl/generate_embeddings.wdl docker=$DOCKER_TAG \
    emb_layer=0 \
    dataset=s3://MYBUCKET/geneformer/datasets/census-2023-10-23/dataset/census-2023-10-23 \
    model=s3://MYBUCKET/geneformer/models/2500_per_cell_type_8epochs/model/2500_per_cell_type_8epochs \
    output_uri=s3://MYBUCKET/geneformer/embs/census-2023-10-23 parts=256 \
    --s3upload s3://MYBUCKET/geneformer/embs
```
