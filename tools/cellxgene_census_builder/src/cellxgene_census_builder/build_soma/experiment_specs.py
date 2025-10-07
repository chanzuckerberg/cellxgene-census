import functools

from .experiment_builder import ExperimentBuilder, ExperimentSpecification
from .globals import (
    ALLOWED_SPATIAL_ASSAYS,
    CXG_OBS_FIELDS_READ,
    CXG_OBS_TERM_FIELDS,
    CXG_OBS_TERM_FIELDS_SPATIAL,
    RNA_SEQ,
)


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
            root_collection="census_data",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9606",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:9606",
            obs_term_fields=CXG_OBS_TERM_FIELDS,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ,
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            label="Mus musculus",
            root_collection="census_data",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:10090",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:10090",
            obs_term_fields=CXG_OBS_TERM_FIELDS,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ,
        ),
        # Experiments for spatial assays
        ExperimentSpecification.create(
            name="homo_sapiens",
            label="Homo sapiens",
            root_collection="census_spatial_sequencing",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9606",
                "assay_ontology_term_ids": ALLOWED_SPATIAL_ASSAYS,
                "is_primary_data": True,
            },
            organism_ontology_term_id="NCBITaxon:9606",
            obs_term_fields=CXG_OBS_TERM_FIELDS + CXG_OBS_TERM_FIELDS_SPATIAL,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ + CXG_OBS_TERM_FIELDS_SPATIAL,
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            label="Mus musculus",
            root_collection="census_spatial_sequencing",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:10090",
                "assay_ontology_term_ids": ALLOWED_SPATIAL_ASSAYS,
                "is_primary_data": True,
            },
            organism_ontology_term_id="NCBITaxon:10090",
            obs_term_fields=CXG_OBS_TERM_FIELDS + CXG_OBS_TERM_FIELDS_SPATIAL,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ + CXG_OBS_TERM_FIELDS_SPATIAL,
        ),
        ExperimentSpecification.create(
            name="callithrix_jacchus",
            label="Callithrix jacchus",
            root_collection="census_data",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9483",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:9483",
            obs_term_fields=CXG_OBS_TERM_FIELDS,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ,
        ),
        ExperimentSpecification.create(
            name="macaca_mulatta",
            label="Macaca mulatta",
            root_collection="census_data",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9544",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:9544",
            obs_term_fields=CXG_OBS_TERM_FIELDS,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ,
        ),
        ExperimentSpecification.create(
            name="pan_troglodytes",
            label="Pan troglodytes",
            root_collection="census_data",
            anndata_cell_filter_spec={
                "organism_ontology_term_id": "NCBITaxon:9598",
                "assay_ontology_term_ids": RNA_SEQ,
            },
            organism_ontology_term_id="NCBITaxon:9598",
            obs_term_fields=CXG_OBS_TERM_FIELDS,
            obs_term_fields_read=CXG_OBS_TERM_FIELDS + CXG_OBS_FIELDS_READ,
        ),
    ]


@functools.cache
def make_experiment_builders() -> list[ExperimentBuilder]:
    return [ExperimentBuilder(spec) for spec in make_experiment_specs()]
