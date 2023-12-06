version development

task finetune_geneformer {
    input {
        Directory dataset
        String output_name

        Directory? model_in
        Int epochs = 8
        File? config

        String docker
    }

    command <<<
        set -euxo pipefail
        >&2 sha256sum /census-geneformer/*.py
        export TQDM_MININTERVAL=10
        python3 /census-geneformer/finetune-geneformer.py \
            -e ~{epochs} ~{'-c ' + config} \
            ~{dataset} \
            ~{select_first([model_in,"/census-geneformer/Geneformer/geneformer-12L-30M"])} \
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
