version development

task finetune_geneformer {
    input {
        Directory dataset
        String output_name
        Int epochs = 8
        File? config
        String docker = "699936264352.dkr.ecr.us-west-2.amazonaws.com/mlin-census-scratch:latest"
    }

    command <<<
        set -euxo pipefail
        >&2 sha256sum /census-geneformer/*.py
        python3 /census-geneformer/finetune-geneformer.py \
            -e ~{epochs} ~{'-c ' + config} \
            ~{dataset} \
            /census-geneformer/Geneformer/geneformer-12L-30M \
            ~{output_name} | tee Trainer.log >&2
    >>>

    output {
        Directory model = output_name
        File trainer_log = "Trainer.log"
    }

    runtime {
        cpu: 48
        memory: "160G"
        gpu: true
        docker: docker
    }
}