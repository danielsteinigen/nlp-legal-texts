import json
import os
from typing import List, Any
from sklearn.model_selection import train_test_split

from util.process_data import Entity, EntityType, Relation, Sample, SampleList
from util.tokenizer import Tokenizer


class DataLoader():

    def __init__(self, data_path: str, spacy_model: str):
        super().__init__()
        self.__tokenizer = Tokenizer(spacy_model)
        self.__data_all = self.__load_set(data_path)
        self.__tokenizer.run(self.__data_all)

    def get_data_all(self) -> SampleList:
        return self.__data_all

    def get_data_split(self, proportion: float) -> (SampleList, SampleList):
        self.__data_train, self.__data_test = train_test_split(self.__data_all.samples, train_size=proportion)
        training_set = SampleList(
            samples=self.__data_train
        )
        test_set = SampleList(
            samples=self.__data_test
        )
        return training_set, test_set

    def __load_set(self, data_path: str) -> SampleList:
        filepath = f"{data_path}"
        with open(filepath, mode='r', encoding="utf-8") as f:
            json_content = json.load(f)

        samples = self.__load_samples(json_content)
        training_set = SampleList(
            samples=samples
        )
        return training_set

    def __load_samples(self, json_content: dict) -> List[Sample]:
        return [self.__load_sample(x) for x in json_content["data"]]

    def __load_sample(self, json_sample) -> Sample:
        entities = []
        relations = []
        if "entities" in json_sample:
            entities = [self.__load_entity(x) for x in json_sample["entities"]]
        if "relations" in json_sample:
            relations = [self.__load_relation(x) for x in json_sample["relations"]]

        return Sample(
            idx=json_sample["id"],
            text=json_sample["text"],
            entities=entities,
            relations=relations
        )

    def __load_entity(self, entity: dict) -> Entity:
        return Entity(
            id=entity["id"],
            text=entity["text"],
            start=entity["start"],
            end=entity["end"],
            ent_type=EntityType(idx=0, label=entity["entity"])
        )

    def __load_relation(self, relation: dict) -> Relation:
        return Relation(
            id=relation["id"],
            head=relation["head"],
            tail=relation["tail"],
            rel_type=EntityType(idx=0, label=relation["relation"]),
        )