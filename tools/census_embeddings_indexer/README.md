# census_embeddings_indexer

This is a Docker+WDL pipeline to build [TileDB-Vector-Search](https://github.com/TileDB-Inc/TileDB-Vector-Search) indexes for Census cell embeddings, supporting cell similarity search in embedding space. It's meant to run on the AWS HealthOmics workflow service using the [miniwdl-omics-run](https://github.com/miniwdl-ext/miniwdl-omics-run) launcher (assuming account setup documented there).

The pipeline consumes one or more of the existing TileDB arrays for hosted and contributed [Census embeddings](https://cellxgene.cziscience.com/census-models) stored on S3. The resulting indexes are themselves TileDB groups to be stored on S3.

```
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_DEFAULT_REGION=$(aws configure get region)
export ECR_ENDPT=${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com

docker build -t ${ECR_ENDPT}/omics:census_embeddings_indexer .
aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_ENDPT"
docker push ${ECR_ENDPT}/omics:census_embeddings_indexer

miniwdl-omics-run census_embeddings_indexer.wdl \
    embeddings_s3_uris=s3_//cellxgene-census-public-us-west-2/cell-census/2023-12-15/soma/census_data/homo_sapiens/ms/RNA/obsm/scvi \
    s3_region=$AWS_DEFAULT_REGION \
    docker=${ECR_ENDPT}/omics:census_embeddings_indexer \
    --output-uri s3://mlin-census-scratch/census_embeddings_indexer/out/ \
    --role poweromics
```

(The `embeddings_s3_uris=s3_//...` is a workaround for an AWS-side existence check that doesn't seem to work right on public buckets.)
