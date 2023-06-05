import logging
import json
from typing import List

from util.process_data import Sample, Entity, EntityType, EntityTypeSet, SampleList
from util.configuration import InferenceConfiguration

import spacy
from spacy.tokens import Doc

class SpacyInference():
   
    def __init__(self, config: InferenceConfiguration):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info(f"Load Configuration: {config.dict()}")

        with open(f"{config.model_path_keyfigure}/classification.json", mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_id_mapping = {x.idx: x for x in self.__entity_type_set.entity_types}

        self.__logger.info("Load Model 1: " + config.model_path_keyfigure)
        self.__model = spacy.load(f"{config.model_path_keyfigure}/model-last")

        self.__second_model = False
        if config.model_path_condition != None and config.model_path_condition != "":
            self.__second_model = True 

            with open(f"{config.model_path_condition}/classification.json", mode='r', encoding="utf-8") as f:
                self.__entity_type_set_2 = EntityTypeSet.parse_obj(json.load(f))
            self.__entity_type_id_mapping_2 = {x.idx: x for x in self.__entity_type_set_2.entity_types}

            self.__logger.info("Load Model 2: " + config.model_path_condition)
            self.__model_2 = spacy.load(f"{config.model_path_condition}/model-last")

        self.__logger.info("Initialization completed.")
       

    def run_inference(self, sample_list: SampleList):

        docs = self.__model.pipe([data.text for data in sample_list.samples])
        for doc, sample in zip(docs, sample_list.samples):
            entities = []
            self.__build_entities(doc, entities, self.__entity_type_id_mapping)
            sample.entities = entities

        if self.__second_model:
            docs_2 = self.__model_2.pipe([data.text for data in sample_list.samples])
            for doc, sample in zip(docs_2, sample_list.samples):
                entities = []
                self.__build_entities(doc, entities, self.__entity_type_id_mapping_2)
                sample.entities += entities

        for sample in sample_list.samples:
            tags = []
            for tok in sample.tokens:
                tag = self.__entity_type_set.id_of_non_entity
                for ent in entities:
                    if tok.start >= ent.start and tok.start < ent.end:
                        tag = ent.ent_type.idx
                tags.append(tag)
            sample.tags = tags


    def __build_entities(self, doc: Doc, entities: List[Entity], entity_type_id_mapping: dict):
        
        for ent in doc.ents:
            entities.append(Entity(
                    id=len(entities)+1,
                    text = ent.text,
                    start = ent.start_char,
                    end = ent.end_char,
                    ent_type=EntityType(
                        idx=int(ent.label_), 
                        label=entity_type_id_mapping[int(ent.label_)].label
                        )
                ))