import logging
import json
from typing import List

from util.process_data import Sample, Entity, EntityType, EntityTypeSet, SampleList
from util.configuration import InferenceConfiguration

import spacy
from spacy.tokens import Token
from rasa.train import train_nlu
from rasa.nlu.model import Interpreter


class RasaInference():
   
    def __init__(self, config: InferenceConfiguration):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info(f"Load Configuration: {config.dict()}")

        with open(f"{config.model_path_keyfigure}/classification.json", mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_id_mapping = {x.idx: x for x in self.__entity_type_set.entity_types}

        self.__logger.info("Load Model 1: " + config.model_path_keyfigure)
        self.__model = Interpreter.load(f"{config.model_path_keyfigure}/nlu")

        self.__second_model = False
        if config.model_path_condition != None and config.model_path_condition != "":
            self.__second_model = True 

            with open(f"{config.model_path_condition}/classification.json", mode='r', encoding="utf-8") as f:
                self.__entity_type_set_2 = EntityTypeSet.parse_obj(json.load(f))
            self.__entity_type_id_mapping_2 = {x.idx: x for x in self.__entity_type_set_2.entity_types}

            self.__logger.info("Load Model 2: " + config.model_path_condition)
            self.__model_2 = Interpreter.load(f"{config.model_path_condition}/nlu")

        self.__logger.info("Initialization completed.")
       

    def run_inference(self, sample_list: SampleList):

        for sample in sample_list.samples:
            entities = []
            prediction = self.__model.parse(sample.text)
            self.__build_entities(prediction, entities, self.__entity_type_id_mapping)
            
            if self.__second_model:
                prediction_2 = self.__model_2.parse(sample.text)
                self.__build_entities(prediction_2, entities, self.__entity_type_id_mapping_2)

            tags = []
            for tok in sample.tokens:
                tag = self.__entity_type_set.id_of_non_entity
                for ent in entities:
                    if tok.start >= ent.start and tok.start < ent.end:
                        tag = ent.ent_type.idx
                tags.append(tag)

            sample.entities = entities
            sample.tags = tags


    def __build_entities(self, prediction: dict, entities: List[Entity], entity_type_id_mapping: dict):
        
        for ent in prediction["entities"]:
            entities.append(Entity(
                id=len(entities)+1,
                text=prediction["text"][ent["start"]:ent["end"]],
                start=ent["start"],
                end=ent["end"],
                ent_type=EntityType(
                    idx=int(ent["entity"]), 
                    label=entity_type_id_mapping[int(ent["entity"])].label
                    ),
                confidence = round(ent["confidence_entity"], 4)
                ))