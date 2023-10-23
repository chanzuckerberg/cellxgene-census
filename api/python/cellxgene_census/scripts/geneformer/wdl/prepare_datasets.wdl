version development

task prepare_census_geneformer_dataset {
    input {
        String output_name

        Array[String] obs_columns = ["soma_joinid", "cell_type", "cell_type_ontology_term_id", "cell_subclass", "cell_subclass_ontology_term_id"]
        Int N = 0
        String census_version = "latest"

        String docker = "699936264352.dkr.ecr.us-west-2.amazonaws.com/mlin-census-scratch:latest"
    }

    command <<<
        set -euxo pipefail
        >&2 sha256sum /census-geneformer/*.py
        mkdir hf
        export HF_HOME="$(pwd)/hf"
        python3 /census-geneformer/prepare-census-geneformer-dataset.py \
            -c '~{sep(",",obs_columns)}' \
            -N ~{N} \
            -v ~{census_version} \
            ~{output_name}
    >>>

    output {
        Directory dataset = output_name
    }

    runtime {
        cpu: 8
        memory: "60G"
        docker: docker
    }
}
