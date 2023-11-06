version development

workflow scatter_generate_embeddings {
    input {
        Directory dataset
        Directory model
        String output_uri
        Int parts = 10

        String docker = "699936264352.dkr.ecr.us-west-2.amazonaws.com/mlin-census-scratch:latest"
    }

    call init_embeddings_array {
        input:
        uri = output_uri, docker
    }

    scatter (part in range(parts)) {
        call generate_embeddings {
            input:
            dataset, model, output_uri, part, parts, docker
        }
    }

    output {}
}

task init_embeddings_array {
    input {
        String uri
        String docker

        Int embedding_dim = 512 # TODO: avoid hard-coding this
    }

    command <<<
        python3 - <<'EOF'
        import tiledbsoma
        import pyarrow as pa
        tiledbsoma.SparseNDArray.create(
            '~{uri}',
            type=pa.float32(),
            shape=(2**31-2, ~{embedding_dim})
        ).close()
        EOF
    >>>

    runtime {
        docker: docker
    }

    output {}
}

task generate_embeddings {
    input {
        Directory dataset
        Directory model
        String output_uri

        # if part is supplied, process only cells satisfying: soma_joinid % parts == part
        Int? part
        Int parts = 1

        String docker
    }

    command <<<
        set -euo pipefail
        >&2 sha256sum /census-geneformer/*.py
        mkdir hf
        export HF_HOME="$(pwd)/hf"
        export TMPDIR="$HF_HOME"
        python3 /census-geneformer/generate-geneformer-embeddings.py \
            ~{"--part " + part} --parts ~{parts} --batch-size 10 --tiledbsoma \
            '~{model}' '~{dataset}' '~{output_uri}'
    >>>

    runtime {
        # sizing to g5.2xlarge since EmbExtractor uses only one GPU
        cpu: 8
        memory: "30G"
        gpu: true
        docker: docker
        # for robustness to sporadic errors e.g.
        # https://github.com/pytorch/pytorch/issues/21819
        maxRetries: 1
    }

    output {}
}
