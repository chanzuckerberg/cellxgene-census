# Census Geneformer training and embeddings generation

These scripts automate:

1. preparing tokenized Geneformer datasets from CELLxGENE Census (`prepare-census-geneformer-dataset.py`)
2. **(deprecated)** fine-tuning a Geneformer cell classifier model (`finetune-geneformer.py`)
3. generate cell embedding vectors given a dataset & model (`generate-geneformer-embeddings.py`)

Embedding generation is computationally intensive on large datasets (e.g. all of Census). To make this practical, a WDL workflow (`wdl/generate_embeddings.wdl`) provides a way to distribute across many compute nodes. The other steps also have WDLs for encapsulation, even though they aren't distributed.

The `Dockerfile` provides the recipe for the docker image used by the WDLs, which packages the scripts together with `cellxgene_census`, Geneformer, pytorch, etc.

(Starting with the 2024-07-01 LTS, [Geneformer includes a model fine-tuned with CELLxGENE](https://huggingface.co/ctheodoris/Geneformer/tree/main/fine_tuned_models/gf-12L-95M-i4096_MTLCellClassifier_CELLxGENE_240522), which we use instead of our own fine-tuning. Our historical fine-tuning code remains here for reference.)

## Example invocations

Using [miniwdl-omics-run](https://github.com/miniwdl-ext/miniwdl-omics-run) for the Amazon HealthOmics workflow service, and assuming the docker image has been built and pushed to ECR (tagged `$DOCKER_TAG`).

Preparing a tokenized dataset for all of Census (>500GB, sharded):

```bash
miniwdl-omics-run wdl/prepare_datasets.wdl \
    docker=$DOCKER_TAG \
    census_version=s3://cellxgene-census-public-us-west-2/cell-census/2025-01-30/soma/ \
    value_filter='is_primary_data==True or is_primary_data==False' \
    output_name=2025-01-30 shards=500 --storage-capacity 4800 \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/datasets/
```

(We set `census_version` to the SOMACollection S3 URI because the HealthOmics workers don't have internet access to the Census release directory endpoint.) The run produces a folder containing 500 shard subfolders named e.g. `shard-123`, under the output URI and HealthOmics run ID.

Generating cell embeddings (takes 8-12h on up to 500 g5.4xlarge, generates 200GB `tiledbsoma.SparseNDArray` on S3):

```bash
seq 0 499 \
    | xargs -n 1 printf 'dataset_shards=s3://MYBUCKET/geneformer/datasets/1234567/out/dataset/2025-01-30/shard-%03d/\n' \
    | xargs -n 9999 miniwdl-omics-run \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/embs \
    wdl/generate_embeddings.wdl \
    docker=$DOCKER_TAG \
    emb_mode=cls emb_layer=0 model_type=Pretrained \
    model=s3://MYBUCKET/geneformer/models/gf-12L-95M-i4096_MTLCellClassifier_CELLxGENE_240522/ \
    output_uri=s3_//MYBUCKET/geneformer/embs/$(date '+%s')/2025-01-30/
```

The `model` input folder can be [copied from upstream](https://huggingface.co/ctheodoris/Geneformer/tree/main/fine_tuned_models/gf-12L-95M-i4096_MTLCellClassifier_CELLxGENE_240522). The `s3_//MYBUCKET` is a workaround for the workflow service rejecting our submission if the specified S3 output folder doesn't yet exist; this workflow creates it using TileDB.

### (deprecated) Fine-tuning procedure

Preparing a tokenized training dataset with 2,500 primary cells per human cell type:

```bash
miniwdl-omics-run wdl/prepare_datasets.wdl \
    docker=$DOCKER_TAG \
    census_version=s3://cellxgene-census-public-us-west-2/cell-census/2023-12-15/soma/ \
    N=2500 sampling_column=cell_type output_name=2500_per_cell_type \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/datasets/
```

Fine-tuning for 8 epochs (takes ~36h on g5.8xlarge):

```bash
miniwdl-omics-run wdl/finetune_geneformer.wdl \
    docker=$DOCKER_TAG \
    dataset=s3://MYBUCKET/geneformer/datasets/2500_per_cell_type/dataset/2500_per_cell_type \
    epochs=8 output_name=2500_per_cell_type_8epochs \
    --role poweromics --output-uri s3://MYBUCKET/geneformer/models/
```

Then the output model folder can be supplied to the `model` input to `generate_embeddings.wdl`.

To change fine-tuning parameters, customize the default `finetune-geneformer.config.yaml` file and supply that to the `config` argument to `finetune_geneformer.wdl`.
