version development

task prepare_census_geneformer_dataset {
    input {
        String output_name

        String value_filter = "is_primary_data==True"
        Array[String] obs_columns = ["soma_joinid", "cell_type", "cell_type_ontology_term_id", "cell_subclass", "cell_subclass_ontology_term_id"]
        Int N = 0
        String sampling_column = "cell_subclass"
        String census_version = "stable"

        String docker
    }

    command <<<
        set -euxo pipefail
        >&2 sha256sum /census-geneformer/*.py
        mkdir hf
        export HF_HOME="$(pwd)/hf"
        export TQDM_MININTERVAL=10
        python3 /census-geneformer/prepare-census-geneformer-dataset.py \
            -c '~{sep(",",obs_columns)}' \
            --value-filter '~{value_filter}' -N ~{N} --sampling-column '~{sampling_column}' \
            -v ~{census_version} \
            ~{output_name}
    >>>

    output {
        Directory dataset = output_name
        File stderr = stderr()
    }

    runtime {
        cpu: 8
        memory: "90G"
        docker: docker
    }
}
