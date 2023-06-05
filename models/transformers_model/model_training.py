import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Any

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch import cuda

from util.evaluation import Score, Evaluation
from util.process_data import Sample, Entity, EntityType, EntityTypeSet, SampleList
from util.configuration import TrainingConfiguration
from transformers_model.config import TrainingParameters


class TokenClassificationDataset(Dataset):
    """ Pytorch Dataset """

    def __init__(self, samples_, tokenizer_, type_set_, align_labels_full_, align_labels_core_, split_sentences_,
                 label_all_tokens_, get_entity_tags_, get_relation_tokens_and_tags_, split_len_):

        self.entity_tokens = [[x.text for x in sample.tokens] for sample in samples_]
        self.entity_tags = get_entity_tags_(samples_)
        self.relation_tokens, self.relation_tags = get_relation_tokens_and_tags_(samples_, use_entity_special_tokens_=True)

        self.tokenizer = tokenizer_
        self.type_set = type_set_
        self.align_labels_core = align_labels_core_
        self.align_labels_full = align_labels_full_
        self.split_sentences = split_sentences_
        self.label_all_tokens = label_all_tokens_

        # split long sentences
        if split_len_ > 0:
            self.entity_tokens, self.entity_tags = self.split_sentences(self.entity_tokens, self.entity_tags, split_len_)
            self.relation_tokens, self.relation_tags = self.split_sentences(self.relation_tokens, self.relation_tags, split_len_)

        self.num_entity_tokens = len(self.entity_tags) * len(self.type_set.groups)


    def __len__(self):
        return self.num_entity_tokens + len(self.relation_tags)


    def __getitem__(self, idx):
        tokens, tags = self.get_filtered_data(idx)
        encodings = self.tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True)

        labels = self.align_labels_core(tags, encodings.word_ids(batch_index=0), self.label_all_tokens)

        item = { key: torch.tensor(val) for key, val in encodings.items() }
        item['labels'] = torch.tensor(labels)
        return item


    def get_filtered_data(self, idx):
        if idx < self.num_entity_tokens:
            # ENTITIES
            grp_idx = idx // len(self.entity_tags)
            par_idx = idx % len(self.entity_tags)
            trigger = f"[GRP-{grp_idx:02d}]"
            tokens = [trigger, *self.entity_tokens[par_idx]]
            #tags = [-100, *self.filter_tags(self.entity_tags[par_idx], grp_idx)]
            tags = [self.type_set.id_of_non_entity, *self.filter_tags(self.entity_tags[par_idx], grp_idx)]
        else:
            # RELATIONS
            rel_idx = idx - self.num_entity_tokens
            tokens = ["[REL]", *self.relation_tokens[rel_idx]]
            #tags = [-100, *self.relation_tags[rel_idx]]
            tags = [self.type_set.id_of_non_entity, *self.relation_tags[rel_idx]]

        return tokens, tags


    def filter_tags(self, tags_, grp_idx_):
        assert grp_idx_ < len(self.type_set.groups)
        res_tags = []
        for tag in tags_:
            res_tag = None
            if isinstance(tag, (list, set, tuple)):
                for subtag in tag:
                    if subtag in self.type_set.groups[grp_idx_]:
                        if res_tag is None:
                            res_tag = subtag
                        else:
                            print(f"[ERROR] multiple tags assigned to a token (last={res_tag} current={subtag} tag={tag} group={grp_idx_}")
                            exit(-11)
            elif tag in self.type_set.groups[grp_idx_]:
                res_tag = tag

            if res_tag is None:
                res_tag = self.type_set.id_of_non_entity

            res_tags.append(res_tag)

        return res_tags


    def get_encodings_and_labels(self):
        data = [self.get_filtered_data(idx) for idx in range(len(self))]
        tokens = [tok for tok, tag in data]
        tags = [tag for tok, tag in data]

        encodings = self.tokenizer(tokens, is_split_into_words=True, padding='max_length', truncation=True)
        labels = self.align_labels_full(tags, encodings, self.label_all_tokens)
        return encodings, labels


class TransformersTraining():

    def __init__(self, training_data: SampleList, test_data: SampleList):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__train_samples = training_data.samples
        self.__test_samples = test_data.samples
        self.__data_path = "transformers_model/model/"

        device = 'cuda' if cuda.is_available() else 'cpu'
        self.__logger.info(f"Device: {device}")
        try:
            self.__logger.info(f"CUDA Device: {cuda.current_device()}")
            self.__logger.info(f"CUDA Number Devices: {cuda.device_count()}")
        except Exception as ex:
            self.__logger.exception("Exception CUDA Device")


    def __setup_configs(self, training_config: TrainingConfiguration):

        self.__logger.info(f"Training parameters used are: {training_config.dict()}")
        with open(training_config.classification_file, mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_label_to_id_mapping = {x.label: x.idx for x in self.__entity_type_set.entity_types}
        self.__entity_type_id_to_label_mapping = {x.idx: x.label for x in self.__entity_type_set.entity_types}

        self.__num_folds = training_config.num_folds
        self.__transformer_model = training_config.transformer_model
        self.__split_len = training_config.split_len
        self.__training_parameters = TrainingParameters()

        self.__tokenizer = AutoTokenizer.from_pretrained(self.__transformer_model,
                padding=self.__training_parameters.padding, max_length=self.__training_parameters.max_sequence_length,
                truncation=self.__training_parameters.truncation)

        self.__logger.info(f"Special tokens (original): '{self.__tokenizer.all_special_tokens}'")

        # add special tokens
        lst_special_tokens = ["[REL]", "[SUB]", "[/SUB]", "[OBJ]", "[/OBJ]"]
        for grp_idx, grp in enumerate(self.__entity_type_set.groups):
            lst_special_tokens.append(f"[GRP-{grp_idx:02d}]")
            lst_special_tokens.extend([f"[ENT-{ent:02d}]" for ent in grp if ent != self.__entity_type_set.id_of_non_entity])
            lst_special_tokens.extend([f"[/ENT-{ent:02d}]" for ent in grp if ent != self.__entity_type_set.id_of_non_entity])

        lst_special_tokens = sorted(list(set(lst_special_tokens)))
        special_tokens_dict = {'additional_special_tokens': lst_special_tokens }
        num_added_toks = self.__tokenizer.add_special_tokens(special_tokens_dict)
        self.__logger.info(f"Added {num_added_toks} new special tokens. All special tokens: '{self.__tokenizer.all_special_tokens}'")

        self.__logger.info("Initialization completed.")


    def run_training(self, training_config: TrainingConfiguration) -> (str, Evaluation):

        self.__setup_configs(training_config)

        k = self.__num_folds
        if k <= 0: # standard training and evaluation
            evaluation = Evaluation()
            model_path = self.__train(training_config, self.__train_samples, self.__test_samples, evaluation, train_ids_=None, val_ids_=None)
            return (model_path, evaluation)

        else:
            all_samples = self.__train_samples + self.__test_samples 
            all_tokens = [[x.text for x in sample.tokens] for sample in all_samples]
            all_tags = self.__get_tags(all_samples)

            kfold = KFold(n_splits=k, shuffle=True)

            res_folds = []
            for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(all_tokens, all_tags)):
                self.__logger.info("-" * 30)
                self.__logger.info(f"Processing fold #{fold_idx+1}")
                self.__logger.info("-" * 30)

                evaluation = Evaluation()
                model_path = self.__train(training_config, all_samples, all_samples, evaluation, train_ids_=train_ids, val_ids_=val_ids)

                res_folds.append({ "idx": fold_idx+1, "train_ids": train_ids.tolist(), "val_ids": val_ids.tolist(),
                                    "eval_results": evaluation, "model_path": model_path })

            return (model_path, self.__get_averaged_scores(res_folds))


    def __build_dataset(self, samples_, indices_=None):
        samples = [samples_[idx] for idx in indices_] if indices_ is not None else samples_[:]
        dataset = TokenClassificationDataset(samples, self.__tokenizer, self.__entity_type_set, self.__align_labels,
                                             self.__align_labels_core, self.__split_sentences, self.__training_parameters.label_all_tokens, self.__get_tags,
                                             self.__get_relation_tokens_and_tags, self.__split_len)

        return dataset


    def __train(self, training_config_, train_samples_, val_samples_, evaluation_, train_ids_=None, val_ids_=None):

        # create directory for model storage
        dt = datetime.now()
        timestamp = f"{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}-{dt.second}"
        output_dir = Path(f"{self.__data_path}{training_config_.model_name}_{timestamp}")
        output_dir.mkdir()

        # store classification file in model folder
        with open(os.path.join(output_dir, "classification.json"), 'w', encoding="utf-8") as f:
            json.dump(self.__entity_type_set.dict(), f, ensure_ascii=False, indent=4)
            
        # build data sets
        train_dataset = self.__build_dataset(train_samples_, indices_=train_ids_)
        val_dataset = self.__build_dataset(val_samples_, indices_=val_ids_)

        model = AutoModelForTokenClassification.from_pretrained(self.__transformer_model, num_labels=len(self.__entity_type_set),
                    max_length=self.__training_parameters.max_sequence_length)

        model.resize_token_embeddings(len(self.__tokenizer))

        training_args = TrainingArguments(
            output_dir = str(output_dir),  # output directory
            num_train_epochs = self.__training_parameters.num_train_epochs,  # total number of training epochs
            per_device_train_batch_size = self.__training_parameters.batch_size,  # batch size per device during training
            per_device_eval_batch_size = self.__training_parameters.batch_size,  # batch size for evaluation
            learning_rate = self.__training_parameters.learning_rate,  # learning rate
            warmup_steps = self.__training_parameters.warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay = self.__training_parameters.weight_decay,  # strength of weight decay
            gradient_accumulation_steps = self.__training_parameters.gradient_accumulation_steps,
        )
        training_args.save_strategy = "no"
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
        )

        trainer.train()

        # Save the model 
        trainer.save_model(str(output_dir))
        self.__tokenizer.save_vocabulary(str(output_dir))
        self.__logger.info("Model saved.")

        if len(val_dataset) == 0:
            self.__logger.warn(f"Validation dataset is empty! Running predictions on the training dataset..")
            self.__run_predictions(trainer, train_dataset, evaluation_)

        else:
            self.__run_predictions(trainer, val_dataset, evaluation_)

        return str(output_dir)


    def __accumulate_score(self, score1_, score2_):
        assert score1_.entity == score2_.entity
        entity_score = Score(
            entity = score1_.entity,
            precision = score1_.precision + score2_.precision,
            recall = score1_.recall + score2_.recall,
            f1 = score1_.f1 + score2_.f1,
            support = score1_.support + score2_.support
        )
        return entity_score


    def __average_score(self, score_, n_, decimal_places_=4):
        assert isinstance(n_, int) and n_ > 0
        score_.precision = round(score_.precision / n_, decimal_places_)
        score_.recall = round(score_.recall / n_, decimal_places_)
        score_.f1 = round(score_.f1 / n_, decimal_places_)
        #score_.support = round(score_.support / n_, decimal_places_)
        return score_


    def __get_averaged_scores(self, results_):
        res = None

        # display results for each fold
        for i, r in enumerate(results_):
            self.__logger.info(f"FOLD {i+1}: {r}")

        # accumulate scores
        for i, r in enumerate(results_):
            v = r["eval_results"]
            if res is None:
                res = v
                counts_for_ents = { ent.entity: 1 for ent in v.score_entities }
            else:
                res.accuracy += v.accuracy
                res.score_macro_avg = self.__accumulate_score(res.score_macro_avg, v.score_macro_avg)
                res.score_weighted_avg = self.__accumulate_score(res.score_weighted_avg, v.score_weighted_avg)

                # accumulate entity scores
                updated_entity_scores = []
                for ent_v in v.score_entities:
                    ent_found = False
                    for ent_res in res.score_entities:
                        if ent_res.entity == ent_v.entity:
                            updated_entity_scores.append(self.__accumulate_score(ent_res, ent_v))
                            counts_for_ents[ent_res.entity] += 1
                            ent_found = True
                            break
                    if not ent_found:
                        updated_entity_scores.append(ent_v)
                        counts_for_ents[ent_v.entity] = 1

                res.score_entities = updated_entity_scores

        # average scores
        assert self.__num_folds > 0
        decimal_places = 4
        res.accuracy = round(res.accuracy / self.__num_folds, decimal_places)
        res.score_macro_avg = self.__average_score(res.score_macro_avg, self.__num_folds, decimal_places)
        res.score_weighted_avg = self.__average_score(res.score_weighted_avg, self.__num_folds, decimal_places)
        res.score_entities = [self.__average_score(ent_score, counts_for_ents[ent_score.entity], decimal_places) for ent_score in res.score_entities]

        return res


    def __split_sentences(self, tokens_, tags_, len_thresh_=200):

        # split token lists & tags into shorter lists
        res_tokens = []
        res_tags = []

        for tok_lst, tag_lst in zip(tokens_, tags_):
            assert len(tok_lst) == len(tag_lst)
            length = len(tok_lst)
            if length > len_thresh_:
                num_lists = length // len_thresh_ + (1 if (length % len_thresh_) > 0 else 0)
                new_length = int(length / num_lists) + 1
                self.__logger.info(f"Splitting a list of {length} elements into {num_lists} lists of length {new_length}..")
                start_idx = 0
                for i in range(num_lists):
                    end_idx = min(start_idx + new_length, length)
                    res_tokens.append(tok_lst[start_idx:end_idx])
                    res_tags.append(tag_lst[start_idx:end_idx])

                    if start_idx > 0 and tag_lst[start_idx] != 0 and tag_lst[start_idx-1] == tag_lst[start_idx]:
                        self.__logger.warn(f"Entity '{tag_lst[start_idx]}' was splitted out (idx={start_idx})!")

                    start_idx = end_idx
            else:
                res_tokens.append(tok_lst)
                res_tags.append(tag_lst)

        return res_tokens, res_tags


    def __run_predictions(self, trainer_, dataset_, evaluation_):

        encodings_, labels_ = dataset_.get_encodings_and_labels()

        # Get predictions of Transformer Sequence Labeling model
        predictions_raw, labels_raw, _ = trainer_.predict(dataset_)

        predictions_align = self.__align_predictions(predictions_raw, encodings_)
        confidence = [[max(token) for token in sample] for sample in predictions_align]
        predictions = [[token.index(max(token)) for token in sample] for sample in predictions_align]

        labels = self.__align_predictions(labels_, encodings_)

        predictions_concat = [j for sub in predictions for j in sub]
        labels_concat = [j for sub in labels for j in sub]

        self.__eval_predictions(labels_concat, predictions_concat, evaluation_)


    def __eval_predictions(self, labels_, predictions_, evaluation_):
        """ evaluate predictions """
        evaluation = classification_report(labels_, predictions_, output_dict=True)

        evaluation_.accuracy = round(evaluation["accuracy"], 4)

        evaluation_.score_macro_avg.precision = round(evaluation["macro avg"]["precision"], 4)
        evaluation_.score_macro_avg.recall = round(evaluation["macro avg"]["recall"], 4)
        evaluation_.score_macro_avg.f1 = round(evaluation["macro avg"]["f1-score"], 4)
        evaluation_.score_macro_avg.support = evaluation["macro avg"]["support"]
        evaluation_.score_weighted_avg.precision = round(evaluation["weighted avg"]["precision"], 4)
        evaluation_.score_weighted_avg.recall = round(evaluation["weighted avg"]["recall"], 4)
        evaluation_.score_weighted_avg.f1 = round(evaluation["weighted avg"]["f1-score"], 4)
        evaluation_.score_weighted_avg.support = evaluation["weighted avg"]["support"]
        for key, value in zip(evaluation.keys(), evaluation.values()):
            if key not in ["accuracy", "macro avg", "weighted avg"]:
                entity = self.__entity_type_id_to_label_mapping[int(key)] if int(key) in self.__entity_type_id_to_label_mapping.keys() else ""
                entity_score = Score(
                    entity=entity,
                    precision=round(value["precision"], 4),
                    recall=round(value["recall"], 4),
                    f1=round(value["f1-score"], 4),
                    support=value["support"])
                evaluation_.score_entities.append(entity_score)


    def __get_tags(self, samples: List[Sample]) -> List[List[int]]:
        overall_tags = []
        for sample in samples:
            tags: List[int] = []
            for token in sample.tokens:
                found_types = []
                for entity in sample.entities:
                    if token.start >= entity.start and token.start < entity.end:
                        entity_type_as_integer = self.__entity_type_label_to_id_mapping[entity.ent_type.label]
                        found_types.append(entity_type_as_integer)
                if len(found_types) == 1:
                    tags.append(found_types[0])
                elif len(found_types) > 1:
                    tags.append(set(found_types))
                else:
                    tags.append(self.__entity_type_set.id_of_non_entity)
            overall_tags.append(tags)
        return overall_tags


    def __get_relation_tokens_and_tags(self, samples: List[Sample], use_entity_special_tokens_ = False) -> List[List[int]]:
        overall_tags, overall_tokens = [], []
        O_tag = self.__entity_type_set.id_of_non_entity

        for sample in samples:
            entity_dict = {entity.id: entity for entity in sample.entities}

            relations = dict()
            for relation in sample.relations:
                head = entity_dict[relation.head]
                tail = entity_dict[relation.tail]
                if not relation.rel_type.label in self.__entity_type_label_to_id_mapping:
                    continue

                rel_type_as_integer = self.__entity_type_label_to_id_mapping[relation.rel_type.label]
                if head.id not in relations:
                    relations[head.id] = (head, [])
                relations[head.id][1].append((tail, rel_type_as_integer))

            head_ids = [relation.head for relation in sample.relations]
            for entity in sample.entities:
                if entity.id not in head_ids:
                    relations[entity.id] = (entity, [(entity, O_tag)])

            for head, tails in relations.values():

                tokens: List[str] = []
                tags: List[int] = []

                last_token_type = None # one of (None, "head", "tail")
                head_entity_type_int = None
                tail_entity_type_int = None
                last_tail_entity_type_int = None
                tail_rel_type = None

                for token in sample.tokens:
                    token_type = None
                    entity_type_int = self.__get_entity_type_int(token, head)

                    if entity_type_int is not None: # IS_HEAD
                        token_type = "head"
                        head_entity_type_int = entity_type_int
                    else:
                        for tail, rel_type_as_int in tails:
                            entity_type_int = self.__get_entity_type_int(token, tail)
                            if entity_type_int is not None: # IS_TAIL
                                token_type = "tail"
                                tail_rel_type = rel_type_as_int
                                tail_entity_type_int = entity_type_int
                                break

                    if token_type not in ["head", "tail"]:
                        for entity in sample.entities:
                            entity_type_int = self.__get_entity_type_int(token, entity)
                            if entity_type_int is not None: # IS_TAIL
                                token_type = "tail"
                                tail_rel_type = O_tag
                                tail_entity_type_int = entity_type_int
                                break

                    if token_type == "head" and last_token_type != "head": # open head
                        if use_entity_special_tokens_:
                            # open head: add [SUB] [ENT-xx] --> TOKENS
                            #            add  --    --      --> TAGS
                            tokens.extend(["[SUB]", f"[ENT-{head_entity_type_int:02d}]"])
                            #tags.extend([-100, -100])
                            tags.extend([O_tag, O_tag])
                        else:
                            # open head: add [SUB] --> TOKENS
                            #            add  --   --> TAGS
                            tokens.append("[SUB]")
                            #tags.append(-100)
                            tags.append(O_tag)
                    elif token_type != "head" and last_token_type == "head": # close head
                        if use_entity_special_tokens_:
                            # close head: add [/ENT-xx] [/SUB] --> TOKENS
                            #             add  --        --    --> TAGS
                            tokens.extend([f"[/ENT-{head_entity_type_int:02d}]", "[/SUB]"])
                            #tags.extend([-100, -100])
                            tags.extend([O_tag, O_tag])
                            head_entity_type = None
                        else:
                            # close head: add [/SUB] --> TOKENS
                            #             add  --    --> TAGS
                            tokens.append("[/SUB]")
                            #tags.append(-100)
                            tags.append(O_tag)

                    if token_type == "tail" and (last_token_type != "tail" or last_tail_entity_type_int != tail_entity_type_int): # open tail
                        if use_entity_special_tokens_:
                            if last_token_type == "tail" and last_tail_entity_type_int != tail_entity_type_int:
                                # close tail: add [/ENT-xx] [/OBJ] --> TOKENS
                                #             add  --        --    --> TAGS
                                tokens.extend([f"[/ENT-{last_tail_entity_type_int:02d}]", "[/OBJ]"])
                                #tags.extend([-100, -100])
                                tags.extend([O_tag, O_tag])
                                #tail_entity_type = None
                            # open tail: add [OBJ] [ENT-xx] --> TOKENS
                            #            add [type] --      --> TAGS
                            tokens.extend(["[OBJ]", f"[ENT-{tail_entity_type_int:02d}]"])
                            #tags.extend([tail_rel_type, -100])
                            tags.extend([tail_rel_type, O_tag])
                        else:
                            if last_token_type == "tail" and last_tail_entity_type_int != tail_entity_type_int:
                                # close tail: add [/OBJ] --> TOKENS
                                #             add  --    --> TAGS
                                tokens.append("[/OBJ]")
                                #tags.append(-100)
                                tags.append(O_tag)
                            # open tail: add [OBJ]  --> TOKENS
                            #            add [type] --> TAGS
                            tokens.append("[OBJ]")
                            tags.append(tail_rel_type)
                        tail_rel_type = None
                    elif token_type != "tail" and last_token_type == "tail": # close tail
                        if use_entity_special_tokens_:
                            # close tail: add [/ENT-xx] [/OBJ] --> TOKENS
                            #             add  --        --    --> TAGS
                            tokens.extend([f"[/ENT-{tail_entity_type_int:02d}]", "[/OBJ]"])
                            #tags.extend([-100, -100])
                            tags.extend([O_tag, O_tag])
                            tail_entity_type = None
                        else:
                            # close tail: add [/OBJ] --> TOKENS
                            #             add  --    --> TAGS
                            tokens.append("[/OBJ]")
                            #tags.append(-100)
                            tags.append(O_tag)

                    last_token_type = token_type
                    last_tail_entity_type_int = tail_entity_type_int

                    tokens.append(token.text)
                    tags.append(self.__entity_type_set.id_of_non_entity)

                overall_tokens.append(tokens)
                overall_tags.append(tags)

        return overall_tokens, overall_tags


    def __get_entity_type_int(self, token_, entity_):
        if token_.start >= entity_.start and token_.start < entity_.end:
            return self.__entity_type_label_to_id_mapping[entity_.ent_type.label]
        else:
            return None


    def __align_predictions(self, predictions, tokenized_inputs, sum_all_tokens=False) -> List[List[List[float]]]:
        """ Align predicted labels from Transformer Tokenizer """
        confidence = []
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


    def __align_labels(self, tags: List[List[int]], tokenized_inputs, label_all_tokens=True) -> list:
        labels = []
        for i, tagset in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            labels.append(self.__align_labels_core(tagset, word_ids, label_all_tokens))

        return labels


    def __align_labels_core(self, tagset, word_ids:List[int], label_all_tokens=True) -> list:

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                tag = tagset[word_idx]
                #if tag == 0:
                #    tag = -100
                label_ids.append(tag)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                tag = tagset[word_idx]
                #if tag == 0:
                #    tag = -100
                label_ids.append(tag if label_all_tokens else -100)

            previous_word_idx = word_idx
        return label_ids
