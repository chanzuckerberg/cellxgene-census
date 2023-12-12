#!/usr/bin/parallel --shebang -j 3 --halt 1 --ungroup
python -m census_contrib ingest-soma --cwd CxG-czi-1 -v --soma-path geneformer-census-2023-10-23 --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-czi-2 -v --joinid-path latent-idx.npy --embedding-path latent.npy --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-czi-3 -v  --joinid-path latent-idx.npy --embedding-path latent.npy --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-contrib-1 -v --joinid-path soma_joinid.txt --embedding-path embeddings.npy --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-contrib-2 -v --joinid-path fixed_human_soma_join_ids.npy --embedding-path cxg_human_2023-10-23_62996451_19715_uce_proc.npy --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-contrib-3 -v --joinid-path cxg_mouse_2023-10-23_soma.npy --embedding-path cxg_mouse_2023-10-23_5684805_22270_uce_proc.npy --census-uri 2023-12-15/soma/
python -m census_contrib ingest-npy --cwd CxG-contrib-4 -v --joinid-path tmp/joinids.npy --embedding-path tmp/h_matrix.npy --census-uri 2023-12-15/soma/
