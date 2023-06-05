from typing import Optional, List

from pydantic import BaseModel, Extra

class TrainingConfiguration(BaseModel):
    model_name: str = "keyfitax_"
    spacy_model: str = "de_core_news_sm"
    transformer_model: str = "xlm-roberta-large"
    classification_file: str = ""
    num_folds: int = 0
    split_len: int = 200


class InferenceConfiguration(BaseModel):
    model_path_keyfigure: str = "rasa_model/model/keyfitax_keyfigure"
    model_path_condition: str = "rasa_model/model/keyfitax_condition"
    spacy_model: str = "de_core_news_sm"
    transformer_model: str = "xlm-roberta-large"
    merge_entities: bool = True
    split_len: int = 200
    extract_relations: bool = True