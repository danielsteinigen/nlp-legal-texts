import json
import logging
from typing import List, Any
import copy

import torch
from injector import inject, singleton
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

from util.process_data import Sample, Entity, EntityType, EntityTypeSet, SampleList, Token, Relation
from util.configuration import InferenceConfiguration

valid_relations = { # head : [tail, ...]
    "Kennzahl": ["Kennzahl", "Bedingung", "Wert", "Wertumschreibung"],
    "Kennzahlumschreibung": ["Kennzahlumschreibung", "Bedingung", "Wert", "Wertumschreibung"],
    "Wert": ["Einheit", "Faktor", "Wertebereich", "Bedingung"],
    "Wertumschreibung": ["Wertumschreibung", "Einheit", "Faktor", "Wertebereich", "Bedingung"],
    "Bedingung": ["Bedingung", "Wert", "Wertumschreibung"],
    "Wertebereich": ["Wertebereich"],
}

class TokenClassificationDataset(Dataset):
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


class TransformersInference():

    def __init__(self, config: InferenceConfiguration):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info(f"Load Configuration: {config.dict()}")

        with open(f"{config.model_path_keyfigure}/classification.json", mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_label_to_id_mapping = {x.label: x.idx for x in self.__entity_type_set.all_types()}
        self.__entity_type_id_to_label_mapping = {x.idx: x.label for x in self.__entity_type_set.all_types()}

        self.__logger.info("Load Model: " + config.model_path_keyfigure)
        self.__tokenizer = AutoTokenizer.from_pretrained(config.transformer_model,
                padding="max_length", max_length=512, truncation=True) 
        
        self.__model = AutoModelForTokenClassification.from_pretrained(config.model_path_keyfigure, num_labels=(
            len(self.__entity_type_set)))

        self.__trainer = Trainer(model=self.__model)
        self.__merge_entities = config.merge_entities
        self.__split_len = config.split_len
        self.__extract_relations = config.extract_relations

        # add special tokens
        entity_groups = self.__entity_type_set.groups
        num_entity_groups = len(entity_groups)

        lst_special_tokens = ["[REL]", "[SUB]", "[/SUB]", "[OBJ]", "[/OBJ]"]
        for grp_idx, grp in enumerate(entity_groups):
            lst_special_tokens.append(f"[GRP-{grp_idx:02d}]")
            lst_special_tokens.extend([f"[ENT-{ent:02d}]" for ent in grp if ent != self.__entity_type_set.id_of_non_entity])
            lst_special_tokens.extend([f"[/ENT-{ent:02d}]" for ent in grp if ent != self.__entity_type_set.id_of_non_entity])

        lst_special_tokens = sorted(list(set(lst_special_tokens)))
        special_tokens_dict = {'additional_special_tokens': lst_special_tokens }
        num_added_toks = self.__tokenizer.add_special_tokens(special_tokens_dict)
        self.__logger.info(f"Added {num_added_toks} new special tokens. All special tokens: '{self.__tokenizer.all_special_tokens}'")

        self.__logger.info("Initialization completed.")



    def run_inference(self, sample_list: SampleList):
        group_predictions = []
        group_entity_ids = []
        self.__logger.info("Predict Entities ...")
        for grp_idx, grp in enumerate(self.__entity_type_set.groups):
            token_lists = [[x.text for x in sample.tokens] for sample in sample_list.samples]
            predictions = self.__get_predictions(token_lists, f"[GRP-{grp_idx:02d}]")
            group_entity_ids_ = []
            for sample, prediction_per_tokens in zip(sample_list.samples, predictions):
                group_entity_ids_.append(self.generate_response_entities(sample, prediction_per_tokens, grp_idx))
            group_predictions.append(predictions)
            group_entity_ids.append(group_entity_ids_)

        if self.__extract_relations:
            self.__logger.info("Predict Relations ...")
            self.__do_extract_relations(sample_list, group_predictions, group_entity_ids)


    def __do_extract_relations(self, sample_list, group_predictions, group_entity_ids):
        id_of_non_entity = self.__entity_type_set.id_of_non_entity

        for sample_idx, sample in enumerate(sample_list.samples):
            masked_tokens = []
            masked_tokens_align = []
            # create SUB-Mask for every entity that can be a head
            head_entities = [entity_ for entity_ in sample.entities if entity_.ent_type.label in list(valid_relations.keys())]
            for entity_ in head_entities:
                ent_masked_tokens = []
                ent_masked_tokens_align = []
                last_preds = [id_of_non_entity for group in group_predictions]
                last_ent_ids = [-1 for group in group_entity_ids]
                for token_idx, token in enumerate(sample.tokens):
                    for group, ent_ids, last_pred, last_ent_id in zip(group_predictions, group_entity_ids, last_preds, last_ent_ids):
                        pred = group[sample_idx][token_idx]
                        ent_id = ent_ids[sample_idx][token_idx]
                        if last_pred != pred and last_pred != id_of_non_entity:
                            mask = "[/SUB]" if last_ent_id == entity_.id else "[/OBJ]"
                            ent_masked_tokens.extend([f"[/ENT-{last_pred:02d}]", mask])
                            ent_masked_tokens_align.extend([str(last_ent_id), str(last_ent_id)])

                    for group, ent_ids, last_pred, last_ent_id in zip(group_predictions, group_entity_ids, last_preds, last_ent_ids):
                        pred = group[sample_idx][token_idx]
                        ent_id = ent_ids[sample_idx][token_idx]
                        if last_pred != pred and pred != id_of_non_entity:
                            mask = "[SUB]" if ent_id == entity_.id else "[OBJ]"
                            ent_masked_tokens.extend([mask, f"[ENT-{pred:02d}]"])
                            ent_masked_tokens_align.extend([str(ent_id), str(ent_id)])

                    ent_masked_tokens.append(token.text)
                    ent_masked_tokens_align.append(token.text)
                    for idx, group in enumerate(group_predictions):
                        last_preds[idx] = group[sample_idx][token_idx]
                    for idx, group in enumerate(group_entity_ids):
                        last_ent_ids[idx] = group[sample_idx][token_idx]

                for group, ent_ids, last_pred, last_ent_id in zip(group_predictions, group_entity_ids, last_preds, last_ent_ids):
                    pred = group[sample_idx][token_idx]
                    ent_id = ent_ids[sample_idx][token_idx]
                    if last_pred != id_of_non_entity:
                        mask = "[/SUB]" if last_ent_id == entity_.id else "[/OBJ]"
                        ent_masked_tokens.extend([f"[/ENT-{last_pred:02d}]", mask])
                        ent_masked_tokens_align.extend([str(last_ent_id), str(last_ent_id)])

                masked_tokens.append(ent_masked_tokens)
                masked_tokens_align.append(ent_masked_tokens_align)

            rel_predictions = self.__get_predictions(masked_tokens, "[REL]")
            self.generate_response_relations(sample, head_entities, masked_tokens_align, rel_predictions)


    def generate_response_entities(self, sample: Sample, predictions_per_tokens: List[int], grp_idx: int):
        entities = []
        entity_ids = []
        id_of_non_entity = self.__entity_type_set.id_of_non_entity
        idx = grp_idx * 1000
        for token, prediction in zip(sample.tokens, predictions_per_tokens):
            if id_of_non_entity == prediction:
                entity_ids.append(-1)
                continue
            idx += 1
            entities.append(self.__build_entity(idx, prediction, token))
            entity_ids.append(idx)

        if self.__merge_entities:
            entities = self.__do_merge_entities(copy.deepcopy(entities))
            prev_pred = id_of_non_entity
            for idx, pred in enumerate(predictions_per_tokens):
                if prev_pred == pred and idx > 0:
                    entity_ids[idx] = entity_ids[idx-1]
                prev_pred = pred

        sample.entities += entities
 
        tags = sample.tags if len(sample.tags) > 0 else [self.__entity_type_set.id_of_non_entity] * len(sample.tokens)
        for tag_id, tok in enumerate(sample.tokens):
            for ent in entities:
                if tok.start >= ent.start and tok.start < ent.end:
                    tags[tag_id] = ent.ent_type.idx
        logging.info(tags)
        sample.tags = tags

        return entity_ids


    def generate_response_relations(self, sample: Sample, head_entities: List[Entity], masked_tokens_align: List[List[str]], rel_predictions: List[List[int]]):
        relations = []
        id_of_non_entity = self.__entity_type_set.id_of_non_entity
        idx = 0
        for entity_, align_per_ent, prediction_per_ent in zip(head_entities, masked_tokens_align, rel_predictions):
            for token, prediction in zip(align_per_ent, prediction_per_ent):
                if id_of_non_entity == prediction:
                    continue
                try:
                    tail = int(token)
                except:
                    continue
                if not self.__validate_relation(sample.entities, entity_.id, tail, prediction):
                    continue
                idx += 1
                relations.append(self.__build_relation(idx, entity_.id, tail, prediction))

        sample.relations = relations


    def __validate_relation(self, entities: List[Entity], head: int, tail: int, prediction: int):
        if head == tail: return False
        head_ents = [ent.ent_type.label for ent in entities if ent.id==head]
        tail_ents = [ent.ent_type.label for ent in entities if ent.id==tail]

        if len(head_ents) > 0:
            head_ent = head_ents[0]
        else:
            return False

        if len(tail_ents) > 0:
            tail_ent = tail_ents[0]
        else:
            return False

        return tail_ent in valid_relations[head_ent]


    def __build_entity(self, idx: int, prediction: int, token: Token) -> Entity:
        return Entity(
            id=idx,
            text=token.text,
            start=token.start,
            end=token.end,
            ent_type=EntityType(
                idx=prediction, 
                label=self.__entity_type_id_to_label_mapping[prediction]
                )
        )

    def __build_relation(self, idx: int, head: int, tail: int, prediction: int) -> Relation:
        return Relation(
            id=idx,
            head=head,
            tail=tail,
            rel_type=EntityType(
                idx=prediction, 
                label=self.__entity_type_id_to_label_mapping[prediction]
                )
        )

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


    def __get_predictions(self, token_lists: List[List[str]], trigger: str) -> List[List[int]]:
        """ Get predictions of Transformer Sequence Labeling model """
        if self.__split_len > 0:
            token_lists_split = self.__do_split_sentences(token_lists, self.__split_len)
            predictions = []
            for sample_token_lists in token_lists_split:
                sample_token_lists_trigger = [[trigger]+sample for sample in sample_token_lists]
                val_encodings = self.__tokenizer(sample_token_lists_trigger, is_split_into_words=True, padding='max_length', truncation=True)  # return_tensors="pt"
                val_labels = []
                for i in range(len(sample_token_lists_trigger)):
                    word_ids = val_encodings.word_ids(batch_index=i)
                    label_ids = [0 for _ in word_ids]
                    val_labels.append(label_ids)

                val_dataset = TokenClassificationDataset(val_encodings, val_labels)

                predictions_raw, _, _ = self.__trainer.predict(val_dataset)

                predictions_align = self.__align_predictions(predictions_raw, val_encodings)
                confidence = [[max(token) for token in sample] for sample in predictions_align]
                predictions_sample = [[token.index(max(token)) for token in sample][1:] for sample in predictions_align]
                predictions_part = []
                for tok, pred in zip(sample_token_lists_trigger, predictions_sample):
                    if trigger == "[REL]" and "[SUB]" not in tok:
                        predictions_part += [self.__entity_type_set.id_of_non_entity] * len(pred)
                    else:
                        predictions_part += pred
                predictions.append(predictions_part)
                # predictions.append([j for i in predictions_sample for j in i]))
        else:
            token_lists_trigger = [[trigger]+sample for sample in token_lists]
            val_encodings = self.__tokenizer(token_lists_trigger, is_split_into_words=True, padding='max_length', truncation=True)  # return_tensors="pt"
            val_labels = []
            for i in range(len(token_lists_trigger)):
                word_ids = val_encodings.word_ids(batch_index=i)
                label_ids = [0 for _ in word_ids]
                val_labels.append(label_ids)

            val_dataset = TokenClassificationDataset(val_encodings, val_labels)

            predictions_raw, _, _ = self.__trainer.predict(val_dataset)

            predictions_align = self.__align_predictions(predictions_raw, val_encodings)
            confidence = [[max(token) for token in sample] for sample in predictions_align]
            predictions = [[token.index(max(token)) for token in sample][1:] for sample in predictions_align]

        return predictions

    def __do_split_sentences(self, tokens_: List[List[str]], split_len_ = 200) -> List[List[List[str]]]:
        # split token lists into shorter lists
        res_tokens = []

        for tok_lst in tokens_:
            res_tokens_sample = []
            length = len(tok_lst)
            if length > split_len_:
                num_lists = length // split_len_ + (1 if (length % split_len_) > 0 else 0)
                new_length = int(length / num_lists) + 1
                self.__logger.info(f"Splitting a list of {length} elements into {num_lists} lists of length {new_length}..")
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
    

    def __align_predictions(self, predictions, tokenized_inputs, sum_all_tokens=False) -> List[List[List[float]]]:
        """ Align predicted labels from Transformer Tokenizer """
        confidence = []
        id_of_non_entity = self.__entity_type_set.id_of_non_entity
        for i, tagset in enumerate(predictions):

            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            token_confidence = []
            for k, word_idx in enumerate(word_ids):
                try:
                    tok_conf = [value for value in tagset[k]]
                except TypeError:
                    # use the object itself it if's not iterable
                    tok_conf = tagset[k]

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
