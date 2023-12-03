# mypy: ignore-errors
"""
Provides classes to recreate cell type and tissue mappings as used in CELLxGENE Discover

- OntologyMapper abstract class to create other mappers
- SystemMapper to map any tissue to a System
- OrganMapper to map any tissue to an Organ
- TissueGeneralMapper to map any tissue to another tissue as shown in Gene Expression and Census
- CellClassMapper to map any cell type to a Cell Class
- CellSubclassMapper to map any cell type to a Cell Subclass

"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import owlready2


class OntologyMapper(ABC):
    # Terms to ignore when mapping
    BLOCK_LIST = [
        "BFO_0000004",
        "CARO_0000000",
        "CARO_0030000",
        "CARO_0000003",
        "NCBITaxon_6072",
        "Thing",
    ]

    def __init__(
        self,
        high_level_ontology_term_ids: List[str],
        ontology_owl_path: Union[str, os.PathLike],
        root_ontology_term_id: str,
    ):
        self._cached_high_level_terms = {}
        self._cached_labels = {}
        self.high_level_terms = high_level_ontology_term_ids
        self.root_ontology_term_id = root_ontology_term_id

        # TODO improve this. First time it loads it raises a TypeError for CL. But redoing it loads it correctly
        # The type error is
        #   'http://purl.obolibrary.org/obo/IAO_0000028' belongs to more than one entity
        #   types (cannot be both a property and a class/an individual)!
        # So we retry only once
        try:
            self._ontology = owlready2.get_ontology(ontology_owl_path).load()
        except TypeError:
            self._ontology = owlready2.get_ontology(ontology_owl_path).load()

    def get_high_level_terms(self, ontology_term_id: str) -> List[Optional[str]]:
        """
        Returns the associated high-level ontology term IDs from any other ID
        """

        ontology_term_id = self.reformat_ontology_term_id(ontology_term_id, to_writable=False)

        if ontology_term_id in self._cached_high_level_terms:
            return self._cached_high_level_terms[ontology_term_id]

        owl_entity = self._get_entity_from_id(ontology_term_id)

        # If not found as an ontology ID  raise
        if not owl_entity:
            raise ValueError("ID not found in the ontology.")

        # List ancestors for this entity, including itself if it is in the list of high level terms
        ancestors = [owl_entity.name] if ontology_term_id in self.high_level_terms else []

        branch_ancestors = self._get_branch_ancestors(owl_entity)
        # Ignore branch ancestors if they are not under the root node
        if branch_ancestors:
            if self.root_ontology_term_id in branch_ancestors:
                ancestors.extend(branch_ancestors)

        # Check if there's at least one top-level entity in the list of ancestors, and add them to
        # the return list of high level term. Always include itself
        resulting_high_level_terms = []
        for high_level_term in self.high_level_terms:
            if high_level_term in ancestors:
                resulting_high_level_terms.append(high_level_term)

        # If no valid high level terms return None
        if len(resulting_high_level_terms) == 0:
            resulting_high_level_terms.append(None)

        resulting_high_level_terms = [
            self.reformat_ontology_term_id(i, to_writable=True) for i in resulting_high_level_terms
        ]
        self._cached_high_level_terms[ontology_term_id] = resulting_high_level_terms

        return resulting_high_level_terms

    def get_top_high_level_term(self, ontology_term_id: str) -> Optional[str]:
        """
        Return the top high level term
        """

        return self.get_high_level_terms(ontology_term_id)[0]

    @abstractmethod
    def _get_branch_ancestors(self, owl_entity):
        """
        Gets ALL ancestors from an owl entity. What's defined as an ancestor depends on the mapper type, for
        example CL ancestors are likely to just include is_a relationship
        """

    def get_label_from_id(self, ontology_term_id: str):
        """
        Returns the label from and ontology term id that is in writable form
        Example: "UBERON:0002048" returns "lung"
        Example: "UBERON_0002048" raises ValueError because the ID is not in writable form
        """

        if ontology_term_id in self._cached_labels:
            return self._cached_labels[ontology_term_id]

        if ontology_term_id is None:
            return None

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

        if ontology_term_id is None:
            return None

        if to_writable:
            if ontology_term_id.count("_") != 1:
                raise ValueError(f"{ontology_term_id} is an invalid ontology term id, it must contain exactly one '_'")
            return ontology_term_id.replace("_", ":")
        else:
            if ontology_term_id.count(":") != 1:
                raise ValueError(f"{ontology_term_id} is an invalid ontology term id, it must contain exactly one ':'")
            return ontology_term_id.replace(":", "_")

    def _list_ancestors(self, entity: owlready2.entity.ThingClass, ancestors: Optional[List[str]] = None) -> List[str]:
        """
        Recursive function that given an entity of an ontology, it traverses the ontology and returns
        a list of all ancestors associated with the entity.
        """
        ancestors = ancestors or []

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

            if entity.name in self.BLOCK_LIST:
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
        return self._ontology.search_one(iri=f"http://purl.obolibrary.org/obo/{ontology_term_id}")

    @staticmethod
    def _is_restriction(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "value")

    @staticmethod
    def _is_entity(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "name")

    @staticmethod
    def _is_and_object(entity: owlready2.entity.ThingClass) -> bool:
        return hasattr(entity, "Classes")


class CellMapper(OntologyMapper):
    # From schema 3.1.0 https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.1.0/schema.md
    CXG_CL_ONTOLOGY_URL = "https://github.com/obophenotype/cell-ontology/releases/download/v2023-07-20/cl.owl"
    # Only look up ancestors under Cell
    ROOT_NODE = "CL_0000000"

    def __init__(self, cell_type_high_level_ontology_term_ids: List[str]):
        super(CellMapper, self).__init__(
            high_level_ontology_term_ids=cell_type_high_level_ontology_term_ids,
            ontology_owl_path=self.CXG_CL_ONTOLOGY_URL,
            root_ontology_term_id=self.ROOT_NODE,
        )

    def _get_branch_ancestors(self, owl_entity):
        branch_ancestors = []
        for is_a in self._get_is_a_for_cl(owl_entity):
            branch_ancestors = self._list_ancestors(is_a, branch_ancestors)

        return set(branch_ancestors)

    @staticmethod
    def _get_is_a_for_cl(owl_entity):
        # TODO make this a recurrent function instead of 2-level for nested loop
        result = []
        for is_a in owl_entity.is_a:
            if CellMapper._is_entity(is_a):
                result.append(is_a)
            elif CellMapper._is_and_object(is_a):
                for is_a_2 in is_a.get_Classes():
                    if CellMapper._is_entity(is_a_2):
                        result.append(is_a_2)

        return result


class TissueMapper(OntologyMapper):
    # From schema 3.1.0 https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.1.0/schema.md
    CXG_UBERON_ONTOLOGY_URL = "https://github.com/obophenotype/uberon/releases/download/v2023-06-28/uberon.owl"

    # Only look up ancestors under anatomical entity
    ROOT_NODE = "UBERON_0001062"

    def __init__(self, tissue_high_level_ontology_term_ids: List[str]):
        self.cell_type_high_level_ontology_term_ids = tissue_high_level_ontology_term_ids
        super(TissueMapper, self).__init__(
            high_level_ontology_term_ids=tissue_high_level_ontology_term_ids,
            ontology_owl_path=self.CXG_UBERON_ONTOLOGY_URL,
            root_ontology_term_id=self.ROOT_NODE,
        )

    def _get_branch_ancestors(self, owl_entity):
        branch_ancestors = []
        for is_a in owl_entity.is_a:
            branch_ancestors = self._list_ancestors(is_a, branch_ancestors)

        return set(branch_ancestors)


class OrganMapper(TissueMapper):
    # List of tissue classes, ORDER MATTERS. If for a given cell type there are multiple cell classes associated
    # then `self.get_top_high_level_term()` returns the one that appears first in th this list
    ORGANS = [
        "UBERON_0000992",  # ovary
        "UBERON_0000029",  # lymph node
        "UBERON_0002048",  # lung
        "UBERON_0002110",  # gallbladder
        "UBERON_0001043",  # esophagus
        "UBERON_0003889",  # fallopian tube
        "UBERON_0018707",  # bladder organ
        "UBERON_0000178",  # blood
        "UBERON_0002371",  # bone marrow
        "UBERON_0000955",  # brain
        "UBERON_0000310",  # breast
        "UBERON_0000970",  # eye
        "UBERON_0000948",  # heart
        "UBERON_0000160",  # intestine
        "UBERON_0002113",  # kidney
        "UBERON_0002107",  # liver
        "UBERON_0000004",  # nose
        "UBERON_0001264",  # pancreas
        "UBERON_0001987",  # placenta
        "UBERON_0002097",  # skin of body
        "UBERON_0002240",  # spinal cord
        "UBERON_0002106",  # spleen
        "UBERON_0000945",  # stomach
        "UBERON_0002370",  # thymus
        "UBERON_0002046",  # thyroid gland
        "UBERON_0001723",  # tongue
        "UBERON_0000995",  # uterus
        "UBERON_0001013",  # adipose tissue
    ]

    def __init__(self):
        super().__init__(tissue_high_level_ontology_term_ids=self.ORGANS)


class SystemMapper(TissueMapper):
    # List of tissue classes, ORDER MATTERS. If for a given cell type there are multiple cell classes associated
    # then `self.get_top_high_level_term()` returns the one that appears first in th this list
    SYSTEMS = [
        "UBERON_0001017",  # central nervous system
        "UBERON_0000010",  # peripheral nervous system
        "UBERON_0001016",  # nervous system
        "UBERON_0001009",  # circulatory system
        "UBERON_0002390",  # hematopoietic system
        "UBERON_0004535",  # cardiovascular system
        "UBERON_0001004",  # respiratory system
        "UBERON_0001007",  # digestive system
        "UBERON_0000922",  # embryo
        "UBERON_0000949",  # endocrine system
        "UBERON_0002330",  # exocrine system
        "UBERON_0002405",  # immune system
        "UBERON_0001434",  # skeletal system
        "UBERON_0000383",  # musculature of body
        "UBERON_0001008",  # renal system
        "UBERON_0000990",  # reproductive system
        "UBERON_0001032",  # sensory system
    ]

    def __init__(self):
        super().__init__(tissue_high_level_ontology_term_ids=self.SYSTEMS)


class TissueGeneralMapper(TissueMapper):
    # List of tissue classes, ORDER MATTERS. If for a given cell type there are multiple cell classes associated
    # then `self.get_top_high_level_term()` returns the one that appears first in th this list
    TISSUE_GENERAL = [
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
        "UBERON_0000014",  # zone of skin
        "UBERON_0000916",  # abdomen
    ]

    def __init__(self):
        super().__init__(tissue_high_level_ontology_term_ids=self.TISSUE_GENERAL)


class CellClassMapper(CellMapper):
    # List of cell classes, ORDER MATTERS. If for a given cell type there are multiple cell classes associated
    # then `self.get_top_high_level_term()` returns the one that appears first in th this list
    CELL_CLASS = [
        "CL_0002494",  # cardiocyte
        "CL_0002320",  # connective tissue cell
        "CL_0000473",  # defensive cell
        "CL_0000066",  # epithelial cell
        "CL_0000988",  # hematopoietic cell
        "CL_0002319",  # neural cell
        "CL_0011115",  # precursor cell
        "CL_0000151",  # secretory cell
        "CL_0000039",  # NEW germ cell line
        "CL_0000064",  # NEW ciliated cell
        "CL_0000183",  # NEW contractile cell
        "CL_0000188",  # NEW cell of skeletal muscle
        "CL_0000219",  # NEW motile cell
        "CL_0000325",  # NEW stuff accumulating cell
        "CL_0000349",  # NEW extraembryonic cell
        "CL_0000586",  # NEW germ cell
        "CL_0000630",  # NEW supporting cell
        "CL_0001035",  # NEW bone cell
        "CL_0001061",  # NEW abnormal cell
        "CL_0002321",  # NEW embryonic cell (metazoa)
        "CL_0009010",  # NEW transit amplifying cell
        "CL_1000600",  # NEW lower urinary tract cell
        "CL_4033054",  # NEW perivascular cell
    ]

    def __init__(self):
        super().__init__(cell_type_high_level_ontology_term_ids=self.CELL_CLASS)


class CellSubclassMapper(CellMapper):
    # List of cell classes, ORDER MATTERS. If for a given cell type there are multiple cell classes associated
    # then `self.get_top_high_level_term()` returns the one that appears first in th this list
    CELL_SUB_CLASS = [
        "CL_0002494",  # cardiocyte
        "CL_0000624",  # CD4-positive, alpha-beta T cell
        "CL_0000625",  # CD8-positive, alpha-beta T cell
        "CL_0000084",  # T cell
        "CL_0000236",  # B cell
        "CL_0000451",  # dendritic cell
        "CL_0000576",  # monocyte
        "CL_0000235",  # macrophage
        "CL_0000542",  # lymphocyte
        "CL_0000738",  # leukocyte
        "CL_0000763",  # myeloid cell
        "CL_0008001",  # hematopoietic precursor cell
        "CL_0000234",  # phagocyte
        "CL_0000679",  # glutamatergic neuron
        "CL_0000617",  # GABAergic neuron
        "CL_0000099",  # interneuron
        "CL_0000125",  # glial cell
        "CL_0000101",  # sensory neuron
        "CL_0000100",  # motor neuron
        "CL_0000117",  # CNS neuron (sensu Vertebrata)
        "CL_0000540",  # neuron
        "CL_0000669",  # pericyte
        "CL_0000499",  # stromal cell
        "CL_0000057",  # fibroblast
        "CL_0000152",  # exocrine cell
        "CL_0000163",  # endocrine cell
        "CL_0000115",  # endothelial cell
        "CL_0002076",  # endo-epithelial cell
        "CL_0002078",  # meso-epithelial cell
        "CL_0011026",  # progenitor cell
        "CL_0000015",  # NEW male germ cell
        "CL_0000021",  # NEW female germ cell
        "CL_0000034",  # NEW stem cell
        "CL_0000055",  # NEW non-terminally differentiated cell
        "CL_0000068",  # NEW duct epithelial cell
        "CL_0000075",  # NEW columnar/cuboidal epithelial cell
        "CL_0000076",  # NEW squamous epithelial cell
        "CL_0000079",  # NEW stratified epithelial cell
        "CL_0000082",  # NEW epithelial cell of lung
        "CL_0000083",  # NEW epithelial cell of pancreas
        "CL_0000095",  # NEW neuron associated cell
        "CL_0000098",  # NEW sensory epithelial cell
        "CL_0000136",  # NEW fat cell
        "CL_0000147",  # NEW pigment cell
        "CL_0000150",  # NEW glandular epithelial cell
        "CL_0000159",  # NEW seromucus secreting cell
        "CL_0000182",  # NEW hepatocyte
        "CL_0000186",  # NEW myofibroblast cell
        "CL_0000187",  # NEW muscle cell
        "CL_0000221",  # NEW ectodermal cell
        "CL_0000222",  # NEW mesodermal cell
        "CL_0000244",  # NEW urothelial cell
        "CL_0000351",  # NEW trophoblast cell
        "CL_0000584",  # NEW enterocyte
        "CL_0000586",  # NEW germ cell
        "CL_0000670",  # NEW primordial germ cell
        "CL_0000680",  # NEW muscle precursor cell
        "CL_0001063",  # NEW neoplastic cell
        "CL_0002077",  # NEW ecto-epithelial cell
        "CL_0002222",  # NEW vertebrate lens cell
        "CL_0002327",  # NEW mammary gland epithelial cell
        "CL_0002503",  # NEW adventitial cell
        "CL_0002518",  # NEW kidney epithelial cell
        "CL_0002535",  # NEW epithelial cell of cervix
        "CL_0002536",  # NEW epithelial cell of amnion
        "CL_0005006",  # NEW ionocyte
        "CL_0008019",  # NEW mesenchymal cell
        "CL_0008034",  # NEW mural cell
        "CL_0009010",  # NEW transit amplifying cell
        "CL_1000296",  # NEW epithelial cell of urethra
        "CL_1000497",  # NEW kidney cell
        "CL_2000004",  # NEW pituitary gland cell
        "CL_2000064",  # NEW ovarian surface epithelial cell
        "CL_4030031",  # NEW interstitial cell
    ]

    def __init__(self, map_orphans_to_class: bool = False):
        if map_orphans_to_class:
            cell_type_high_level = self.CELL_SUB_CLASS + CellClassMapper.CELL_CLASS
        else:
            cell_type_high_level = self.CELL_SUB_CLASS
        super().__init__(cell_type_high_level_ontology_term_ids=cell_type_high_level)
