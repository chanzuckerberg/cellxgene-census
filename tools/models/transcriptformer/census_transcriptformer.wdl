version development

workflow census_transcriptformer {
    input {
        String output_uri

        String organism = "Homo sapiens"
        String model = "tf_sapiens"
        String census_version = "2025-01-30"
        String? value_filter
        Int? mod
        Int? batch_size
        Int? megabatch_size
        Int? embedding_dim

        Int shards = 1

        String s3_region = "us-west-2"
        String docker

        # The following scripts are baked into the Docker image, but can be overridden here (for
        # faster dev iteration w/o re-CodeBuilding the image)
        File? planner_py
        File? inference_driver_py
        File? put_embeddings_py
    }

    # work around any tooling that might try to verify pre-existence of the output URI when
    # launching the workflow:
    String output_uri2 = sub(output_uri, "s3_//", "s3://")
    String census_uri = "s3://cellxgene-census-public-us-west-2/cell-census/~{census_version}/soma/"

    # plan sharding and create the output TileDB array
    call prepare {
        input:
        census_uri, organism, value_filter, mod, shards,
        output_uri = output_uri2, s3_region, embedding_dim,
        docker, planner_py
    }

    # for each shard, generate embeddings and write them into the output array
    scatter (plan_json in prepare.plans_json) {
        call generate_embeddings {
            input:
            plan_json, model,
            output_uri = output_uri2, s3_region, docker,
            batch_size, megabatch_size,
            inference_driver_py, put_embeddings_py
        }
    }

    output {}
}

task prepare {
    input {
        String census_uri
        String organism
        String? value_filter
        Int? mod
        Int shards

        String output_uri
        String s3_region
        Int embedding_dim = 2048

        String docker
        File? planner_py
    }

    String? s_mod = if defined(mod) then "~{mod}" else None

    command <<<
        set -euo pipefail

        if [[ -n '~{planner_py}' ]]; then
            cp '~{planner_py}' /census-transcriptformer
        fi
        >&2 sha256sum /census-transcriptformer/*.py

        python3 /census-transcriptformer/planner.py \
            --census-uri '~{census_uri}' --organism '~{organism}' \
            ~{"--value-filter '" + value_filter + "'"} \
            ~{"--mod " + s_mod} \
            --shards '~{shards}'

        # create the output TileDB array, using venv with pinned, older tiledbsoma/tiledb
        # to ensure compatibility with Census embeddings curator
        /census-transcriptformer/embeddings_tiledbsoma_venv/bin/python - <<'EOF'
        import tiledb
        import tiledbsoma
        import pyarrow as pa
        tiledbsoma.SparseNDArray.create(
            '~{output_uri}',
            type=pa.float32(),
            shape=(2**31-2, ~{embedding_dim}),
            context = tiledbsoma.options.SOMATileDBContext(
                tiledb_config={"vfs.s3.region": '~{s3_region}'}
            )
        ).close()
        EOF
    >>>

    runtime {
        docker: docker
    }

    output {
        # plan files written by planner.py
        Array[File] plans_json = glob("plan_*.json")
    }
}

task generate_embeddings {
    input {        
        File plan_json
        String output_uri
        String s3_region
        String docker

        String model
        Int? batch_size
        Int? megabatch_size

        File? inference_driver_py
        File? put_embeddings_py
    }

    String? s_batch_size = if defined(batch_size) then "~{batch_size}" else None
    String? s_megabatch_size = if defined(megabatch_size) then "~{megabatch_size}" else None

    command <<<
        set -euxo pipefail
        if [[ -n '~{inference_driver_py}' ]]; then
            cp '~{inference_driver_py}' /census-transcriptformer
        fi
        if [[ -n '~{put_embeddings_py}' ]]; then
            cp '~{put_embeddings_py}' /census-transcriptformer
        fi
        >&2 sha256sum /census-transcriptformer/*.py
        export AWS_DEFAULT_REGION='~{s3_region}'
        export TQDM_MININTERVAL=10
        python3 /census-transcriptformer/inference_driver.py \
            ~{"--megabatch-size " + s_megabatch_size} \
            ~{"--batch-size " + s_batch_size} \
            --model '~{model}' \
            '~{plan_json}' '~{output_uri}'
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
