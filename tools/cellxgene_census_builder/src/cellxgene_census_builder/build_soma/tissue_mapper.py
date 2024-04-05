"""NOTE: In the future, this code will be part of an ontology service library."""

from cellxgene_ontology_guide import curated_ontology_term_lists
from cellxgene_ontology_guide.entities import CuratedOntologyTermList
from cellxgene_ontology_guide.ontology_parser import OntologyParser


class TissueMapper:
    def __init__(self) -> None:
        self.ontology_parser = OntologyParser()
        self.tissues = curated_ontology_term_lists.get_curated_ontology_term_list(
            CuratedOntologyTermList.TISSUE_GENERAL
        )

    def get_high_level_tissue(self, tissue_ontology_term_id: str) -> str:
        """Returns the associated high-level tissue ontology term ID from any other ID.

        Edge cases:
            - If multiple high-level tissues exists for a given tissue, returns the one with higher priority (the first
            appearance in list self.HIGH_LEVEL_TISSUES.
            - If no high-level tissue is found, returns the same as input.
            - If the input tissue is not found in the ontology, return the same as input.
                - This could happen with something like "UBERON:0002048 (cell culture)"
        """
        tissues: list[str] = self.ontology_parser.get_high_level_terms(tissue_ontology_term_id, self.tissues)
        if not tissues:
            raise ValueError()
        return tissues[0]

    def get_label_from_writable_id(self, ontology_term_id: str) -> str:
        """Returns the label from and ontology term id that is in writable form.

        Example: "UBERON:0002048" returns "lung"
        Example: "UBERON_0002048" raises ValueError because the ID is not in writable form
        """
        return self.ontology_parser.get_term_label(ontology_term_id)

    @staticmethod
    def reformat_ontology_term_id(ontology_term_id: str, to_writable: bool = True) -> str:
        """Converts ontology term id string between two formats.

        - `to_writable == True`: from "UBERON_0002048" to "UBERON:0002048"
        - `to_writable == False`: from "UBERON:0002048" to "UBERON_0002048"
        """
        if to_writable:
            if ontology_term_id.count("_") != 1:
                raise ValueError(f"{ontology_term_id} is an invalid ontology term id, it must contain exactly one '_'")
            return ontology_term_id.replace("_", ":")
        else:
            if ontology_term_id.count(":") != 1:
                raise ValueError(f"{ontology_term_id} is an invalid ontology term id, it must contain exactly one ':'")
            return ontology_term_id.replace(":", "_")
