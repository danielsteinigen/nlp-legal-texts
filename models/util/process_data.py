from typing import Optional, List

from pydantic import BaseModel, Extra

class EntityType(BaseModel):
    idx: int
    label: str


class EntityTypeSet(BaseModel):
    entity_types: List[EntityType]
    relation_types: List[EntityType]
    id_of_non_entity: int
    groups: List[List[int]]

    def __len__(self):
        return len(self.entity_types) + len(self.relation_types)

    def all_types(self):
        return [*self.entity_types, *self.relation_types]


class Token(BaseModel):
    text: str
    start: int
    end: int


class Entity(BaseModel):
    id: int
    text: str
    start: int
    end: int
    ent_type: EntityType 
    confidence: Optional[float]


class Relation(BaseModel):
    id: int
    head: int
    tail: int
    rel_type: EntityType


class Sample(BaseModel):
    idx: int
    text: str
    entities: List[Entity] = []
    relations: List[Relation] = []
    tokens: List[Token] = []
    tags: List[int] = []


class SampleList(BaseModel):
    samples: List[Sample]
