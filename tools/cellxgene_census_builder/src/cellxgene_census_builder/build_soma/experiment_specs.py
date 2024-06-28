import functools

from .experiment_builder import ExperimentBuilder, ExperimentSpecification
from .globals import ALLOWED_SPATIAL_ASSAYS, RNA_SEQ


@functools.cache
def make_experiment_specs() -> list[ExperimentSpecification]:
    """Define all soma.Experiments to build in the census.

    Functionally, this defines per-experiment name, anndata filter, etc.
    It also loads any required per-Experiment assets.
    """
    return [  # The soma.Experiments we want to build
        ExperimentSpecification.create(
            name="homo_sapiens",
            label="Homo sapiens",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9606",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:9606",
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            label="Mus musculus",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:10090",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:10090",
        ),
        # Experiments for spatial assays
        ExperimentSpecification.create(
            name="homo_sapiens",
            label="Homo sapiens",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9606",
                "assay_ontology_term_ids": ALLOWED_SPATIAL_ASSAYS,
            },
            organism_ontology_term_id="NCBITaxon:9606",
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            label="Mus musculus",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:10090",
                "assay_ontology_term_ids": ALLOWED_SPATIAL_ASSAYS,
            },
            organism_ontology_term_id="NCBITaxon:10090",
        ),
    ]


@functools.cache
def make_experiment_builders() -> list[ExperimentBuilder]:
    return [ExperimentBuilder(spec) for spec in make_experiment_specs()]
