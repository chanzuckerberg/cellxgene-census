# type: ignore
# isort:skip_file
# flake8: noqa
"""
NOTE: This is a (literal) copy of
https://github.com/chanzuckerberg/single-cell-data-portal/blob/9b94ccb0a2e0a8f6182b213aa4852c491f6f6aff/backend/wmg/data/tissue_mapper.py

Please do not modify this file directly here. Instead, modify the original file in single-cell-data-portal, run the unit tests (which exist in that repo),
get the PR approved and merged, and then port back the changes to this file.

In the future, this code will be part of an ontology service library.

This code contains several places that do not pass the lint/static analysis CI for this pipeline, so the analysis is disabled in this prologue.
"""
from typing import List

import owlready2

from .globals import CXG_UBERON_ONTOLOGY_URL


class TissueMapper:
    # Name of anatomical structure, used to determine the set of ancestors for a given
    # entity that we"re interested in.
    ANATOMICAL_STRUCTURE_NAME = "UBERON_0000061"

    # List of high level tissues, ORDER MATTERS. If for a given tissue there are multiple high-level tissues associated
    # then `self.get_high_level_tissue()` returns the one that appears first in th this list
    HIGH_LEVEL_TISSUES = [
        "UBERON_0000178",  # blood
        "UBERON_0002048",  # lung
        "UBERON_0002106",  # spleen
        "UBERON_0002371",  # bone marrow
        "UBERON_0002107",  # liver
        "UBERON_0002113",  # kidney
        "UBERON_0000955",  # brain
        "UBERON_0002240",  # spinal cord
        "UBERON_0000310",  # breast
        "UBERON_0000948",  # heart
        "UBERON_0002097",  # skin of body
        "UBERON_0000970",  # eye
        "UBERON_0001264",  # pancreas
        "UBERON_0001043",  # esophagus
        "UBERON_0001155",  # colon
        "UBERON_0000059",  # large intestine
        "UBERON_0002108",  # small intestine
        "UBERON_0000160",  # intestine
        "UBERON_0000945",  # stomach
        "UBERON_0001836",  # saliva
        "UBERON_0001723",  # tongue
        "UBERON_0001013",  # adipose tissue
        "UBERON_0000473",  # testis
        "UBERON_0002367",  # prostate gland
        "UBERON_0000057",  # urethra
        "UBERON_0000056",  # ureter
        "UBERON_0003889",  # fallopian tube
        "UBERON_0000995",  # uterus
        "UBERON_0000992",  # ovary
        "UBERON_0002110",  # gall bladder
        "UBERON_0001255",  # urinary bladder
        "UBERON_0018707",  # bladder organ
        "UBERON_0000922",  # embryo
        "UBERON_0004023",  # ganglionic eminence --> this a part of the embryo, remove in case generality is desired
        "UBERON_0001987",  # placenta
        "UBERON_0007106",  # chorionic villus
        "UBERON_0002369",  # adrenal gland
        "UBERON_0002368",  # endocrine gland
        "UBERON_0002365",  # exocrine gland
        "UBERON_0000030",  # lamina propria
        "UBERON_0000029",  # lymph node
        "UBERON_0004536",  # lymph vasculature
        "UBERON_0001015",  # musculature
        "UBERON_0000004",  # nose
        "UBERON_0003688",  # omentum
        "UBERON_0000977",  # pleura
        "UBERON_0002370",  # thymus
        "UBERON_0002049",  # vasculature
        "UBERON_0009472",  # axilla
        "UBERON_0001087",  # pleural fluid
        "UBERON_0000344",  # mucosa
        "UBERON_0001434",  # skeletal system
        "UBERON_0002228",  # rib
        "UBERON_0003129",  # skull
        "UBERON_0004537",  # blood vasculature
        "UBERON_0002405",  # immune system
        "UBERON_0001009",  # circulatory system
        "UBERON_0001007",  # digestive system
        "UBERON_0001017",  # central nervous system
        "UBERON_0001008",  # renal system
        "UBERON_0000990",  # reproductive system
        "UBERON_0001004",  # respiratory system
        "UBERON_0000010",  # peripheral nervous system
        "UBERON_0001032",  # sensory system
        "UBERON_0002046",  # thyroid gland
        "UBERON_0004535",  # cardiovascular system
        "UBERON_0000949",  # endocrine system
        "UBERON_0002330",  # exocrine system
        "UBERON_0002390",  # hematopoietic system
        "UBERON_0000383",  # musculature of body
        "UBERON_0001465",  # knee
        "UBERON_0001016",  # nervous system
        "UBERON_0001348",  # brown adipose tissue
        "UBERON_0015143",  # mesenteric fat pad
        "UBERON_0000175",  # pleural effusion
        "UBERON_0001416",  # skin of abdomen
        "UBERON_0001868",  # skin of chest
        "UBERON_0001511",  # skin of leg
        "UBERON_0002190",  # subcutaneous adipose tissue
        "UBERON_0035328",  # upper outer quadrant of breast
        "UBERON_0000014",  # zone of skin
    ]

    # Terms to ignore when mapping
    DENY_LIST = [
        "BFO_0000004",
        "CARO_0000000",
        "CARO_0030000",
        "CARO_0000003",
        "NCBITaxon_6072",
        "Thing",
        "UBERON_0000465",  # material anatomical entity
        "UBERON_0001062",  # anatomical entity
    ]

    def __init__(self):
        self._cached_tissues = {}
        self._cached_labels = {}
        self._uberon = owlready2.get_ontology(CXG_UBERON_ONTOLOGY_URL).load()

    def get_high_level_tissue(self, tissue_ontology_term_id: str) -> str:
        """
        Returns the associated high-level tissue ontology term ID from any other ID
        Edge cases:
            - If multiple high-level tissues exists for a given tissue, returns the one with higher priority (the first
            appearance in list self.HIGH_LEVEL_TISSUES.
            - If no high-level tissue is found, returns the same as input.
            - If the input tissue is not found in the ontology, return the same as input.
                - This could happen with something like "UBERON:0002048 (cell culture)"
        """

        tissue_ontology_term_id = self.reformat_ontology_term_id(tissue_ontology_term_id, to_writable=False)

        if tissue_ontology_term_id in self._cached_tissues:
            # If we have looked this up already
            return self._cached_tissues[tissue_ontology_term_id]

        entity = self._get_entity_from_id(tissue_ontology_term_id)

        if not entity:
            # If not found as an ontology ID return itself
            result = self.reformat_ontology_term_id(tissue_ontology_term_id, to_writable=True)
            self._cached_tissues[tissue_ontology_term_id] = result
            return result

        # List ancestors for this entity, including itself. Ignore any ancestors that
        # are not descendents of UBERON_0000061 (anatomical structure).
        ancestors = [entity.name]
        branch_ancestors = []
        for is_a in entity.is_a:
            branch_ancestors = self._list_ancestors(is_a, branch_ancestors)

        # Include this branch of ancestors is under anatomical structure
        if self.ANATOMICAL_STRUCTURE_NAME in branch_ancestors:
            ancestors.extend(branch_ancestors)

        # Check if there's at least one top-level entity in the list of ancestors
        # for this entity
        selected_tissue = tissue_ontology_term_id
        for high_level_tissue in self.HIGH_LEVEL_TISSUES:
            if high_level_tissue in ancestors:
                selected_tissue = high_level_tissue
                break

        result = self.reformat_ontology_term_id(selected_tissue, to_writable=True)
        self._cached_tissues[tissue_ontology_term_id] = result
        return result

    def get_label_from_writable_id(self, ontology_term_id: str):
        """
        Returns the label from and ontology term id that is in writable form
        Example: "UBERON:0002048" returns "lung"
        Example: "UBERON_0002048" raises ValueError because the ID is not in writable form
        """

        if ontology_term_id in self._cached_labels:
            return self._cached_labels[ontology_term_id]

        entity = self._get_entity_from_id(self.reformat_ontology_term_id(ontology_term_id, to_writable=False))
        if entity:
            result = entity.label[0]
        else:
            result = ontology_term_id

        self._cached_labels[ontology_term_id] = result
        return result

    @staticmethod
    def reformat_ontology_term_id(ontology_term_id: str, to_writable: bool = True):
        """
        Converts ontology term id string between two formats:
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

    def _list_ancestors(self, entity: owlready2.entity.ThingClass, ancestors: List[str] = []) -> List[str]:
        """
        Recursive function that given an entity of an ontology, it traverses the ontology and returns
        a list of all ancestors associated with the entity.
        """

        if self._is_restriction(entity):
            # Entity is a restriction, check for part_of relationship

            prop = entity.property.name
            if prop != "BFO_0000050":
                # BFO_0000050 is "part of"
                return ancestors
            ancestors.append(entity.value.name.replace("obo.", ""))

            # Check for ancestors of restriction
            self._list_ancestors(entity.value, ancestors)
            return ancestors

        elif self._is_entity(entity) and not self._is_and_object(entity):
            # Entity is a superclass, check for is_a relationships

            if entity.name in self.DENY_LIST:
                return ancestors
            ancestors.append(entity.name)

            # Check for ancestors of superclass
            for super_entity in entity.is_a:
                self._list_ancestors(super_entity, ancestors)
            return ancestors

    def _get_entity_from_id(self, ontology_term_id: str) -> owlready2.entity.ThingClass:
        """
        Given a readable ontology term id (e.g. "UBERON_0002048"), it returns the associated ontology entity
        """
        return self._uberon.search_one(iri=f"http://purl.obolibrary.org/obo/{ontology_term_id}")

    @staticmethod
    def _is_restriction(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "value")

    @staticmethod
    def _is_entity(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "name")

    @staticmethod
    def _is_and_object(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "Classes")
