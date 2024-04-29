version development

workflow census_embeddings_indexer {
    input {
        # S3 folder URIs e.g.
        # s3_//cellxgene-census-public-us-west-2/cell-census/2023-12-15/soma/census_data/homo_sapiens/ms/RNA/obsm/scvi
        Array[String] embeddings_s3_uris        
        String s3_region = "us-west-2"
        String docker
    }

    scatter (embeddings_s3_uri in embeddings_s3_uris) {
        call indexer {
            input:
            embeddings_s3_uri, s3_region, docker
        }
    }

    output {
        Array[Directory] indexes = indexer.index
    }
}

task indexer {
    input {
        String embeddings_s3_uri
        String s3_region
        String embeddings_name = basename(embeddings_s3_uri)
        Int partitions = 100

        String docker
        Int cpu = 16
    }

    command <<<
        set -euxo pipefail

        python3 << 'EOF'
        import tiledb
        import tiledb.vector_search as vs

        config = tiledb.Config().dict()
        config["vfs.s3.region"] = "~{s3_region}"

        source_uri = "~{embeddings_s3_uri}".replace("s3_//", "s3://")
        with tiledb.open(source_uri, config=config) as emb_array:
            emb_shape = emb_array.shape

        vs.ingest(
            config=config,
            source_uri=source_uri,
            source_type="TILEDB_SPARSE_ARRAY",
            dimensions=emb_shape[1],
            index_type="IVF_FLAT",
            index_uri="./~{embeddings_name}",
            partitions=~{partitions},
            training_sampling_policy=vs.ingestion.TrainingSamplingPolicy.RANDOM
        )

        with vs.ivf_flat_index.IVFFlatIndex(uri="./~{embeddings_name}", memory_budget=1024*1048756) as final_index:
            assert final_index.size == emb_shape[0]
        EOF
    >>>

    runtime {
        cpu: cpu
        memory: "~{2*cpu}G"
        docker: docker
    }

    output {
        Directory index = "~{embeddings_name}"
    }
}
