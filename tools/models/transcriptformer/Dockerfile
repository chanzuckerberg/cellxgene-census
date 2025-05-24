FROM ubuntu:24.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
RUN pip3 install --break-system-packages \
        transcriptformer==0.2.1 \
        cellxgene-census
WORKDIR /app
RUN transcriptformer download tf-sapiens
#RUN transcriptformer download tf-exemplar
#RUN transcriptformer download tf-metazoa
COPY driver.py .

# TODO: venv for put_embeddings
