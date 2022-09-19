#!/usr/bin/env python3
import json
import os
from typing import List, Any
import copy

import spacy
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

from types_extractor import Entity, EntityType, EntityTypeSet, Token, Sample


class CFG:
    spacy_model = "de_core_news_sm"
    transformer_model = "xlm-roberta-large"
    model_location = "models/kiss_kennzahl"
    model_2_location = "models/kiss_bedingung"
    classification_file = "models/kiss_kennzahl/classification.json"
    classification_2_file = "models/kiss_bedingung/classification.json"


class ExtractorDataset(Dataset):
    """ Pytorch Dataset """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class EntityExtractor():

    def __init__(self):
        super().__init__()
        print("Load Configuration... ")
        self.__spacy_model = spacy.load(CFG.spacy_model)
        
        with open(CFG.classification_file, mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_id_mapping = {x.idx: x for x in self.__entity_type_set.entity_types}

        with open(CFG.classification_2_file, mode='r', encoding="utf-8") as f:
            self.__entity_type_set_2 = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_id_mapping_2 = {x.idx: x for x in self.__entity_type_set_2.entity_types}
        
        self.__language_model_tokenizer = AutoTokenizer.from_pretrained(CFG.transformer_model, padding="max_length", max_length=512, truncation=True) 
        
        print("Load Model 1: ")
        self.__model = AutoModelForTokenClassification.from_pretrained(CFG.model_location, num_labels=(len(self.__entity_type_set.entity_types)))
        self.__trainer = Trainer(model=self.__model)

        print("Load Model 2: " )
        self.__model_2 = AutoModelForTokenClassification.from_pretrained(CFG.model_2_location, num_labels=(len(self.__entity_type_set_2.entity_types)))
        self.__trainer_2 = Trainer(model=self.__model_2)

        print("Initialization completed.")

    def __tokenize(self, samples: List[Sample]):
        doc_pipe = self.__spacy_model.pipe([sample.text for sample in samples])
        for sample, doc in zip(samples, doc_pipe):
            sample.tokens = [Token(
                text=x.text,
                start=x.idx,
                end=x.idx + len(x.text)
            ) for x in doc]

    def run_inference(self, samples: List[Sample]):
        self.__tokenize(samples)

        predictions = self.__get_predictions(samples, model_no=1)
        for sample, prediction_per_tokens in zip(samples, predictions):
            self.generate_response(sample, prediction_per_tokens, model_no=1)

        predictions_2 = self.__get_predictions(samples, model_no=2)
        for sample, prediction_per_tokens in zip(samples, predictions_2):
            self.generate_response(sample, prediction_per_tokens, model_no=2)
        

    def generate_response(self, sample: Sample, predictions_per_tokens: List[int], model_no: int):
        entities = []
        id_of_non_entity = self.__entity_type_set_2.id_of_non_entity if model_no == 2 else self.__entity_type_set.id_of_non_entity
        for token, prediction in zip(sample.tokens, predictions_per_tokens):
            if id_of_non_entity == prediction:
                continue
            entities.append(self.__build_entity(prediction, token, model_no))
        
        entities = self.__do_merge_entities(copy.deepcopy(entities))

        sample.entities += entities
        sample.tags = predictions_per_tokens if model_no != 2 else sample.tags

    def __do_merge_entities(self, input_ents_):
        out_ents = list()
        current_ent = None

        for ent in input_ents_:
            if current_ent is None:
                current_ent = ent
            else:
                idx_diff = ent.start - current_ent.end
                if ent.ent_type.idx == current_ent.ent_type.idx and idx_diff <= 1:
                    current_ent.end = ent.end
                    current_ent.text += (" " if idx_diff == 1 else "") + ent.text
                else:
                    out_ents.append(current_ent)
                    current_ent = ent
        
        if current_ent is not None:
            out_ents.append(current_ent)

        return out_ents

    def __build_entity(self, prediction: int, token: Token, model_no: int) -> Entity:
        return Entity(
            text=token.text,
            start=token.start,
            end=token.end,
            ent_type=EntityType(
                idx=prediction, 
                label=self.__entity_type_id_mapping_2[prediction].label if model_no == 2 else self.__entity_type_id_mapping[prediction].label
                )
        )


    def __get_predictions(self, samples: List[Sample], model_no: int) -> List[List[int]]:
        """ Get predictions of Transformer Sequence Labeling model """
        token_lists = [[x.text for x in sample.tokens] for sample in samples]
        token_lists_split = self.__do_split_sentences(token_lists)
        predictions = []
        for sample_token_lists in token_lists_split:
            val_encodings = self.__language_model_tokenizer(sample_token_lists, is_split_into_words=True, padding=True,
                                                            truncation=True)  # return_tensors="pt"
            print("Length Token encodings: "+str(len(val_encodings["input_ids"][0])))
            val_labels = []
            for i in range(len(sample_token_lists)):
                word_ids = val_encodings.word_ids(batch_index=i)
                label_ids = [0 for _ in word_ids]
                val_labels.append(label_ids)

            val_dataset = ExtractorDataset(val_encodings, val_labels)

            predictions_raw, _, _ = self.__trainer_2.predict(val_dataset) if model_no == 2 else self.__trainer.predict(val_dataset)

            predictions_align = self.__align_predictions(predictions_raw, val_encodings, model_no)
            confidence = [[max(token) for token in sample] for sample in predictions_align]
            predictions_sample = [[token.index(max(token)) for token in sample] for sample in predictions_align]
            predictions.append([j for i in predictions_sample for j in i])

        return predictions

    def __do_split_sentences(self, tokens_: List[List[str]], len_thresh_ = 200) -> List[List[List[str]]]:

        # split token lists into shorter lists
        res_tokens = []

        for tok_lst in tokens_:
            res_tokens_sample = []
            length = len(tok_lst)
            if length > len_thresh_:
                num_lists = length // len_thresh_ + (1 if (length % len_thresh_) > 0 else 0)
                new_length = int(length / num_lists) + 1
                print(f"Splitting a list of {length} elements into {num_lists} lists of length {new_length}..")
                start_idx = 0
                for i in range(num_lists):
                    end_idx = min(start_idx + new_length, length)
                    if "\n" in tok_lst[start_idx]: tok_lst[start_idx] = "."
                    if "\n" in tok_lst[end_idx-1]: tok_lst[end_idx-1] = "."
                    res_tokens_sample.append(tok_lst[start_idx:end_idx])
                    start_idx = end_idx

                res_tokens.append(res_tokens_sample)
            else:
                res_tokens.append([tok_lst])

        return res_tokens
    

    def __align_predictions(self, predictions, tokenized_inputs, model_no: int, sum_all_tokens=False) -> List[List[List[float]]]:
        """ Align predicted labels from Transformer Tokenizer """
        confidence = []
        id_of_non_entity = self.__entity_type_set_2.id_of_non_entity if model_no == 2 else self.__entity_type_set.id_of_non_entity
        for i, tagset in enumerate(predictions):

            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            token_confidence = []
            for k, word_idx in enumerate(word_ids):
                tok_conf = [value for value in tagset[k]]

                if word_idx is not None:

                    # add nonentity tokens if there is a gap in word ids (usually caused by a newline token)
                    if previous_word_idx is not None:
                        diff = word_idx - previous_word_idx
                        for i in range(diff - 1):
                            tmp = [0 for _ in tok_conf]
                            tmp[id_of_non_entity] = 1.0
                            token_confidence.append(tmp)

                    # add confidence value if this is the first token of the word
                    if word_idx != previous_word_idx:
                        token_confidence.append(tok_conf)
                    else:
                        # if sum_all_tokens=True the confidence for all tokens of one word will be summarized
                        if sum_all_tokens:
                            token_confidence[-1] = [a + b for a, b in zip(token_confidence[-1], tok_conf)]

                previous_word_idx = word_idx

            confidence.append(token_confidence)

        return confidence
