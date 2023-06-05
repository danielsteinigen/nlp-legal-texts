from typing import List

from pydantic import BaseModel


class Score(BaseModel):
    entity: str = ""
    f1: float = .0
    precision: float = .0
    recall: float = .0
    support: int = 0


class Evaluation(BaseModel):
    score_entities: List[Score] = []
    score_macro_avg: Score = Score()
    score_weighted_avg: Score = Score()
    accuracy: float = .0
