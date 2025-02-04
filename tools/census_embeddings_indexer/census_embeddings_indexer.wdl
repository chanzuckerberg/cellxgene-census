version development

workflow census_embeddings_indexer {
    input {
        # S3 folder URIs e.g.
        # s3_//cellxgene-census-public-us-west-2/cell-census/2023-12-15/soma/census_data/homo_sapiens/ms/RNA/obsm/scvi
        Array[String] embeddings_s3_uris        
        String census_version
        String s3_region = "us-west-2"
        String docker
    }

    scatter (embeddings_s3_uri in embeddings_s3_uris) {
        call indexer {
            input:
            embeddings_s3_uri, s3_region, docker
        }
    }

    call make_one_directory {
        input:
        directories = indexer.index,
        directory_name = census_version,
        docker
    }

    output {
        Directory indexes = make_one_directory.directory
    }
}

task indexer {
    input {
        String embeddings_s3_uri
        String s3_region
        String embeddings_name = basename(embeddings_s3_uri)

        String docker
        Int cpu = 32
    }

    command <<<
        set -euxo pipefail

        python3 << 'EOF'
        import sys
        import math
        import tiledb
        import tiledb.vector_search as vs

        config = tiledb.Config().dict()
        config["vfs.s3.region"] = "~{s3_region}"

        source_uri = "~{embeddings_s3_uri}".replace("s3_//", "s3://")
        with tiledb.open(source_uri, config=config) as emb_array:
            (_, N), (_, M) = emb_array.nonempty_domain() # FIXME should be "current domain"
        N += 1  # ASSUMES contiguous soma_joinid's [0, N)
        M += 1
        input_vectors_per_work_item = 1_500_000_000 // M  # controls memory usage
        print(f"N={N} M={M} input_vectors_per_work_item={input_vectors_per_work_item}", file=sys.stderr)

        vs.ingest(
            config=config,
            source_uri=source_uri,
            source_type="TILEDB_SPARSE_ARRAY",
            size=N,
            dimensions_override=M,  # FIXME: see Dockerfile
            index_type="IVF_FLAT",
            index_uri="./~{embeddings_name}",
            partitions=math.ceil(math.sqrt(N)),
            training_sampling_policy=vs.ingestion.TrainingSamplingPolicy.RANDOM,
            input_vectors_per_work_item=input_vectors_per_work_item,
            input_vectors_per_work_item_during_sampling=input_vectors_per_work_item,
            verbose=True,
        )

        final_index = vs.ivf_flat_index.IVFFlatIndex(uri="./~{embeddings_name}", memory_budget=1024*1048756)
        print(f"VACUUM", file=sys.stderr)
        final_index.vacuum()
        assert final_index.size == N, f"final_index.size=={final_index.size} != N=={N}"
        EOF

        >&2 ls -lR '~{embeddings_name}'
    >>>

    runtime {
        cpu: cpu
        memory: "~{8*cpu}G"
        docker: docker
    }

    output {
        Directory index = "~{embeddings_name}"
    }
}

task make_one_directory {
    input {
        Array[Directory] directories
        String directory_name = "indexes"
        String docker
    }

    File manifest = write_lines(directories)

    command <<<
        set -euxo pipefail
        mkdir -p '~{directory_name}'
        while read -r dir; do
            cp -r "$dir" '~{directory_name}/'
        done < '~{manifest}'
        >&2 ls -lR '~{directory_name}'
    >>>

    output {
        Directory directory = directory_name
    }

    runtime {
        docker: docker
        memory: "4G"
    }
}
