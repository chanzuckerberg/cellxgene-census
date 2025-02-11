# census_embeddings_indexer

This is a Docker+WDL pipeline to build [TileDB-Vector-Search](https://github.com/TileDB-Inc/TileDB-Vector-Search) indexes for Census cell embeddings, supporting cell similarity search in embedding space. It's meant to run on the AWS HealthOmics workflow service using the [miniwdl-omics-run](https://github.com/miniwdl-ext/miniwdl-omics-run) launcher (`pip3 install miniwdl-omics-run`; one-time account setup steps documented there are probably already done in the relevant CZI AWS account).

The pipeline consumes one or more of the existing TileDB arrays for [Census embeddings](https://cellxgene.cziscience.com/census-models) stored on S3. The resulting indexes are themselves TileDB groups to be stored on S3.

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_DEFAULT_REGION=$(aws configure get region)
export ECR_ENDPT=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com
export WDL_OUTPUT_BUCKET=mlin-census-scratch

docker build -t ${ECR_ENDPT}/omics:census_embeddings_indexer .
aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_ENDPT"
docker push ${ECR_ENDPT}/omics:census_embeddings_indexer

miniwdl-omics-run census_embeddings_indexer.wdl \
    embeddings_s3_uris=s3_//cellxgene-contrib-public/contrib/cell-census/soma/2024-07-01/CxG-czi-6 \
    embeddings_s3_uris=s3_//cellxgene-contrib-public/contrib/cell-census/soma/2024-07-01/CxG-czi-7 \
    embeddings_s3_uris=s3_//cellxgene-contrib-public/contrib/cell-census/soma/2024-07-01/CxG-czi-8 \
    embeddings_s3_uris=s3_//cellxgene-contrib-public/contrib/cell-census/soma/2024-07-01/CxG-contrib-7 \
    census_version=2024-07-01 \
    s3_region=$AWS_DEFAULT_REGION \
    docker=${ECR_ENDPT}/omics:census_embeddings_indexer \
    --output-uri s3://${WDL_OUTPUT_BUCKET}/census_embeddings_indexer/out/ \
    --role poweromics --storage-capacity 4800
```

(The `embeddings_s3_uris=s3_//...` with `s3_//` instead of `s3://` is a workaround for an AWS-side existence check that doesn't seem to work right on public buckets.)

The [Dockerfile](Dockerfile) has arguments for the TileDB-Py and TileDB-Vector-Search versions to use; see comments there for guidance on setting them.
