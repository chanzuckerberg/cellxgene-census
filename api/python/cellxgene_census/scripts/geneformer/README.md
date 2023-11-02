# Census Geneformer scripts

These scripts automate:

1. preparing a tokenized Geneformer dataset from the Census (`prepare-census-geneformer-dataset.py`)
2. fine-tuning a Geneformer cell classifier model (`finetune-geneformer.py`)
3. using the fine-tuned model to generate cell embedding vectors (`generate-geneformer-embeddings.py`)

Embedding generation is computationally intensive to run on large datasets (e.g. all of Census). To enable this, a WDL workflow (`wdl/generate_embeddings.wdl`) provides a way to distribute this across many compute nodes. There are also WDLs for the other steps, to help with encapsulating them even though they aren't distributed.

The `Dockerfile` provides the recipe for the docker image used by the WDLs, which packages the scripts together with `cellxgene_census`, Geneformer, pytorch, etc.

## Example invocations

Using a [miniwdl-aws-terraform](https://github.com/miniwdl-ext/miniwdl-aws-terraform) deployment with suitable GPU instance types enabled, and assuming the docker image has been built and pushed to a suitable repository like ECR.

```bash
# generating subsampled dataset
miniwdl-aws-submit prepare_datasets.wdl \
    N=1250 sampling_column=cell_type output_name=1250_per_cell_type \
    --verbose --follow --s3upload s3://BUCKET/FOLDER

# generating full Census dataset
miniwdl-aws-submit prepare_datasets.wdl output_name=census-YYYY-MM-DD \
    --verbose --follow --s3upload s3://BUCKET/FOLDER

# fine-tuning on g5.48xlarge
MINIWDL__AWS__GPU_VALUE=8
MINIWDL__AWS__CONTAINER_PROPERTIES='{"linuxParameters":{"sharedMemorySize":4096}}'
miniwdl-aws-submit finetune_geneformer.wdl \
    dataset=s3://BUCKET/DATASET_FOLDER \
    epochs=10 output_name=MyGeneformerClassifier \
    --verbose --follow --s3upload s3://BUCKET/FOLDER

# generating embeddings on ten g5.2xlarge
miniwdl-aws-submit generate_embeddings.wdl \
    dataset=s3://BUCKET/DATASET_FOLDER \
    model=s3://BUCKET/MODEL_FOLDER \
    output_name=MyEmbeddings parts=10 \
    --verbose --follow --s3upload s3://BUCKET/FOLDER
```
