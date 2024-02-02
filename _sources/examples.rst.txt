Python tutorials
==========


Exporting data
----------

Learn how to stream the single-cell data and metadata from Census into your machine.

.. toctree::
    :maxdepth: 1
   
    cellxgene_census_docsite_quick_start.md
    notebooks/api_demo/census_query_extract.ipynb
    notebooks/api_demo/census_citation_generation.ipynb
    notebooks/api_demo/census_gget_demo.ipynb

[NEW! 🚀] Using integrated embeddings and models
----------

Tutorials that show you how to retrieve pre-calculated Census embeddings and use their associated models for your workflows.

Access Census embeddings.

.. toctree::
    :maxdepth: 1
   
    notebooks/api_demo/census_access_maintained_embeddings.ipynb
    notebooks/api_demo/census_embedding.ipynb
    
Use the Census trained models.

.. toctree::
    :maxdepth: 1
    
    notebooks/analysis_demo/comp_bio_scvi_model_use.ipynb 
    notebooks/analysis_demo/comp_bio_geneformer_prediction.ipynb
   
Exploring human biology with Census embeddings.

.. toctree::
    :maxdepth: 1
    
    notebooks/analysis_demo/comp_bio_embedding_exploration.ipynb

Understanding Census data
----------

Gain a better understanding on the nature of the Census data and how it's organized.

.. toctree::
    :maxdepth: 1
    
    notebooks/analysis_demo/census_duplicated_cells.ipynb
    notebooks/analysis_demo/comp_bio_census_info.ipynb
    notebooks/analysis_demo/census_summary_cell_counts.ipynb
    notebooks/analysis_demo/comp_bio_explore_and_load_lung_data.ipynb
    notebooks/api_demo/census_datasets.ipynb
    notebooks/api_demo/census_dataset_presence.ipynb
    notebooks/api_demo/census_summary_cell_counts.ipynb
   
Analyzing Census data
----------
   
A few examples of relevant analysis pipelines with Census data.
   
.. toctree::
    :maxdepth: 1
   
    notebooks/analysis_demo/comp_bio_summarize_axis_query.ipynb
    notebooks/analysis_demo/comp_bio_data_integration_scvi.ipynb
    notebooks/analysis_demo/comp_bio_normalizing_full_gene_sequencing.ipynb

Scalable computing
----------

Demonstrations of memory-efficient compute workflows that leverage the streaming capabilities of Census.

.. toctree::
    :maxdepth: 1
    
    notebooks/experimental/highly_variable_genes.ipynb
    notebooks/experimental/mean_variance.ipynb
    notebooks/analysis_demo/census_compute_over_X.ipynb

Scalable machine learning
----------

Learn about features to do data modeling directly from Census into machine learning toolkits.

.. toctree::
    :maxdepth: 1
    
    notebooks/experimental/pytorch.ipynb
