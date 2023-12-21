import functools
from typing import List

from .experiment_builder import ExperimentBuilder, ExperimentSpecification
from .globals import RNA_SEQ


@functools.cache
def make_experiment_specs() -> List[ExperimentSpecification]:
    """
    Define all soma.Experiments to build in the census.

    Functionally, this defines per-experiment name, anndata filter, etc.
    It also loads any required per-Experiment assets.
    """
    return [  # The soma.Experiments we want to build
        ExperimentSpecification.create(
            name="homo_sapiens",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:9606", assay_ontology_term_ids=RNA_SEQ),
        ),
        ExperimentSpecification.create(
            name="mus_musculus",
            anndata_cell_filter_spec=dict(organism_ontology_term_id="NCBITaxon:10090", assay_ontology_term_ids=RNA_SEQ),
        ),
    ]


@functools.cache
def make_experiment_builders() -> List[ExperimentBuilder]:
    return [ExperimentBuilder(spec) for spec in make_experiment_specs()]
