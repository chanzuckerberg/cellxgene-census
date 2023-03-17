from typing import List

from .experiment_builder import ExperimentSpecification
from .globals import RNA_SEQ


def make_experiment_specs() -> List[ExperimentSpecification]:
    """
    Define all soma.Experiments to build in the census.

    Functionally, this defines per-experiment name, anndata filter, etc.
    It also loads any required per-Experiment assets.
    """
    GENE_LENGTH_BASE_URI = (
        "https://raw.githubusercontent.com/chanzuckerberg/single-cell-curation/"
        "100f935eac932e1f5f5dadac0627204da3790f6f/cellxgene_schema_cli/cellxgene_schema/ontology_files/"
    )
    GENE_LENGTH_URIS = [
        GENE_LENGTH_BASE_URI + "genes_homo_sapiens.csv.gz",
        GENE_LENGTH_BASE_URI + "genes_mus_musculus.csv.gz",
        GENE_LENGTH_BASE_URI + "genes_sars_cov_2.csv.gz",
    ]
    return [  # The soma.Experiments we want to build
        ExperimentSpecification.create(
            name="homo_sapiens",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:9606", assay_ontology_term_ids=RNA_SEQ),
            gene_feature_length_uris=GENE_LENGTH_URIS,
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:10090", assay_ontology_term_ids=RNA_SEQ),
            gene_feature_length_uris=GENE_LENGTH_URIS,
        ),
    ]
