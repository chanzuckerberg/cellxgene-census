version development

workflow scatter_generate_embeddings {
    input {
        Directory dataset
        Directory model
        String output_name
        String? label_feature       
        Int parts = 10

        String? docker
    }

    scatter (part in range(parts)) {
        call generate_embeddings {
            input:
            dataset, model, output_name, label_feature, part, parts, docker
        }
    }

    # TODO: gather, following scvi example

    output {
        Array[File] embeddings = generate_embeddings.embeddings
    }
}

task generate_embeddings {
    input {
        Directory dataset
        Directory model
        String output_name

        String label_feature = "cell_subclass"

        # if part is supplied, process only cells satisfying: soma_joinid % parts == part
        Int? part
        Int parts = 1

        String docker = "699936264352.dkr.ecr.us-west-2.amazonaws.com/mlin-census-scratch:latest"
    }

    String outfile = if part == None then "~{output_name}.tsv" else "~{output_name}.~{part}.tsv"

    command <<<
        set -euo pipefail
        >&2 sha256sum /census-geneformer/*.py
        mkdir hf
        export HF_HOME="$(pwd)/hf"
        python3 /census-geneformer/generate-geneformer-embeddings.py \
            ~{"--part " + part} --parts ~{parts} --label-feature '~{label_feature}' \
            '~{model}' '~{dataset}' '~{outfile}'
    >>>

    runtime {
        cpu: 48
        memory: "160G"
        gpu: true
        docker: docker
    }

    output {
        File embeddings = outfile
    }
}
