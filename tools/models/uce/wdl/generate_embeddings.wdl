version development

workflow scatter_generate_embeddings {
    input {
        String output_uri = "s3://pablo-tmp-west-coast-2/uce/emb_test/"
        String s3_region = "us-west-2"
        Int parts = 1000
        Int emb_dim = 1280 #TODO not hardcode
        Directory? model_directory
        String docker
        #String docker = "uce"
    }

    # work around any tooling that might try to verify pre-existence of the output URI when
    # launching the workflow:
    String output_uri2 = sub(output_uri, "s3_//", "s3://")

    # create the output TileDB array and move model files to right location
    call init_embeddings_array {
        input:
        uri = output_uri2, s3_region, docker, emb_dim
    }

    # generate each shard's embeddings and write them into the above-created array
    scatter (part in range(1)) {
        call generate_embeddings after init_embeddings_array {
            input:
            uri = output_uri2, s3_region, docker, emb_dim, part, parts, model_directory
        }
    }

    output {}
}

task init_embeddings_array {
    input {
        String uri
        String s3_region
        String docker
        Int emb_dim
    }

    command <<<
        set -euo pipefail
        source /census-uce/embeddings_tiledbsoma_venv/bin/activate
        export AWS_DEFAULT_REGION='~{s3_region}'
        python3 /census-uce/prepare-tiledbsoma-array.py \
            --emb-dim '~{emb_dim}' --s3-region '~{s3_region}' '~{uri}'
        set -euo pipefail
    >>>

    runtime {
        docker: docker
    }

    output {}
}

task generate_embeddings {
    input {
        String uri
        String s3_region
        String docker
        Int emb_dim
        Int part
        Int parts
        Directory? model_directory
    }

    command <<<
        set -euo pipefail
        export AWS_DEFAULT_REGION='~{s3_region}'
        cd /census-uce/
        ln -s '~{model_directory}'/* ./UCE/model_files/
        python3 ./generate-uce-embeddings.py \
            --part '~{part}' --parts '~{parts}' --emb-dim '~{emb_dim}' \
            --tiledbsoma --remove-h5ads \
            --output-dir-census tmp_census --output-dir tmp_census_uce \
            '~{uri}'
        set -euo pipefail
    >>>

    runtime {
        # sizing to g5.2xlarge since EmbExtractor uses only one GPU
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
    
    #requirements {
    #   gpu: true
    #}
    #
    #hints {
    #   gpu: 1
    #}
}
