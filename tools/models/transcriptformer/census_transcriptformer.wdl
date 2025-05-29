version development

workflow census_transcriptformer {
    input {
        String output_uri

        String census_version = "2025-01-30"
        String organism = "Homo sapiens"
        String? value_filter
        Int? mod
        Int? batch_size
        Int? megabatch_size
        Int? embedding_dim

        String s3_region = "us-west-2"
        String docker
    }

    # work around any tooling that might try to verify pre-existence of the output URI when
    # launching the workflow:
    String output_uri2 = sub(output_uri, "s3_//", "s3://")

    # create the output TileDB array
    call prepare {
        input:
        uri = output_uri2, s3_region, docker, embedding_dim
    }

    # generate embeddings and write them into the above-created array
#    scatter (shard in dataset_shards) {
        call generate_embeddings after prepare {
            input:
            output_uri = output_uri2,
            s3_region, docker,
            census_version, organism,
            value_filter, mod,
            batch_size, megabatch_size
        }
#    }

    output {}
}

task prepare {
    input {
        String uri
        String s3_region
        String docker

        Int embedding_dim = 2048
    }

    command <<<
        set -euo pipefail
        # use venv with pinned, older tiledbsoma/tiledb to ensure compatibility with
        # Census embeddings curator
        source /census-transcriptformer/embeddings_tiledbsoma_venv/bin/activate
        python3 - <<'EOF'
        import tiledb
        import tiledbsoma
        import pyarrow as pa
        tiledbsoma.SparseNDArray.create(
            '~{uri}',
            type=pa.float32(),
            shape=(2**31-2, ~{embedding_dim}),
            context = tiledbsoma.options.SOMATileDBContext(
                tiledb_config={"vfs.s3.region": '~{s3_region}'}
            )
        ).close()
        EOF

        # TODO: detect total cell count of census_version/organism, and generate
        # Array[Pair[Int,Int]] of id ranges
    >>>

    runtime {
        docker: docker
    }

    output {}
}

task generate_embeddings {
    input {        
        String output_uri
        String s3_region
        String docker

        String census_version = "2025-01-30"
        String organism = "Homo sapiens"
        String? value_filter
        Int? mod
        Int? batch_size
        Int? megabatch_size
        # TODO: lo/hi
    }

    String? s_mod = if defined(mod) then "~{mod}" else None
    String? s_batch_size = if defined(batch_size) then "~{batch_size}" else None
    String? s_megabatch_size = if defined(megabatch_size) then "~{megabatch_size}" else None

    command <<<
        set -euo pipefail
        >&2 sha256sum /census-transcriptformer/*.py
        export AWS_DEFAULT_REGION='~{s3_region}'
        export TQDM_MININTERVAL=10
        python3 /census-transcriptformer/driver.py \
            ~{"--obs-value-filter '" + value_filter + "'"} \
            ~{"--obs-mod " + s_mod} \
            ~{"--megabatch-size " + s_megabatch_size} \
            ~{"--batch-size " + s_batch_size} \
            --census-uri 's3://cellxgene-census-public-us-west-2/cell-census/~{census_version}/soma/' \
            --organism '~{organism}' \
            --array '~{output_uri}'
    >>>

    runtime {
        # sizing to g5.2xlarge
        cpu: 8
        memory: "30G"
        gpu: true
        acceleratorCount: 1
        acceleratorType: "nvidia-tesla-a10g"
        docker: docker
        # for robustness to sporadic errors e.g.
        # https://github.com/pytorch/pytorch/issues/21819
        maxRetries: 1
    }

    output {}
}
