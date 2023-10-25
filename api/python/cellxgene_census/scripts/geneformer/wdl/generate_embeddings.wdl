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

    scatter (which_part in range(parts)) {
        call generate_embeddings {
            input:
            dataset, model, output_name, label_feature, which_part, parts, docker
        }
    }

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

        # if which_part is supplied, process only cells satisfying: soma_joinid % parts == which_part
        Int? which_part
        Int parts = 1

        String docker = "699936264352.dkr.ecr.us-west-2.amazonaws.com/mlin-census-scratch:latest"
    }

    String outfile = if which_part == None then "~{output_name}.tsv" else "~{output_name}.~{which_part}.tsv"

    command <<<
        set -euo pipefail
        >&2 sha256sum /census-geneformer/*.py
        mkdir hf
        export HF_HOME="$(pwd)/hf"
        python3 /census-geneformer/generate-geneformer-embeddings.py \
            ~{"--part " + which_part} --parts ~{parts} --label-feature '~{label_feature}' \
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
