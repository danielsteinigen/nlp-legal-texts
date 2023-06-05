import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from util.evaluation import Score, Evaluation
from util.process_data import Sample, Entity, EntityType, EntityTypeSet, SampleList
from util.configuration import TrainingConfiguration

import spacy
from spacy.tokens import Doc, DocBin
from spacy.cli.train import train
from spacy.util import filter_spans


class SpacyTraining():

    def __init__(self, training_data: SampleList, test_data: SampleList):
        super().__init__()
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__train_samples = training_data.samples
        self.__test_samples = test_data.samples
        self.__data_path = "spacy_model/model/"
        self.__model_config = "spacy_model/config.cfg"


    def run_training(self, training_config: TrainingConfiguration) -> (str, Evaluation):
        self.__logger.info(f"Training parameters used are: {training_config.dict()}")
        with open(training_config.classification_file, mode='r', encoding="utf-8") as f:
            self.__entity_type_set = EntityTypeSet.parse_obj(json.load(f))
        self.__entity_type_label_to_id_mapping = {x.label: x.idx for x in self.__entity_type_set.entity_types}
        self.__entity_type_id_to_label_mapping = {x.idx: x.label for x in self.__entity_type_set.entity_types}

        self.__num_folds = training_config.num_folds
        self.__spacy_model = spacy.load(training_config.spacy_model)

        self.__logger.info("Initialization completed.")
   
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


    def __train(self, training_config, train_samples_, val_samples_, evaluation_, train_ids_=None, val_ids_=None):

        # create directory for model storage
        dt = datetime.now()
        timestamp = f"{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}-{dt.second}"
        output_dir = Path(f"{self.__data_path}{training_config.model_name}_{timestamp}")
        output_dir.mkdir()

        # store classification file in model folder
        with open(os.path.join(output_dir, "classification.json"), 'w', encoding="utf-8") as f:
            json.dump(self.__entity_type_set.dict(), f, ensure_ascii=False, indent=4)
            
        # build data sets
        train_dataset = [train_samples_[idx] for idx in train_ids_] if train_ids_ is not None else train_samples_[:]
        val_dataset = [val_samples_[idx] for idx in val_ids_] if val_ids_ is not None else val_samples_[:]
        self.__convert_data_format(train_dataset, "train")
        self.__convert_data_format(val_dataset, "test")

        # train model
        train(config_path=self.__model_config, output_path=str(output_dir), overrides={"paths.train": self.__data_path+"train.spacy", "paths.dev": self.__data_path+"test.spacy"}) # use_gpu=1
        self.__logger.info("Model saved.")

        self.__run_predictions(output_dir, val_dataset, evaluation_)

        return str(output_dir)


    def __run_predictions(self, output_dir, dataset_, evaluation_):
        # Get predictions of trained model
        model_predict = spacy.load(str(os.path.join(output_dir, "model-last")))

        predicted_data = []
        docs = model_predict.pipe([sample.text for sample in dataset_])
        for doc, sample in zip(docs, dataset_):
            entities = []
            for ent_id, ent in enumerate(doc.ents):
                entities.append(Entity(
                    id=ent_id,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    ent_type=EntityType(
                        idx=int(ent.label_), 
                        label=self.__entity_type_id_to_label_mapping[int(ent.label_)]
                        )
                    ))
            predicted_data.append(Sample(
                idx=sample.idx,
                text=sample.text,
                tokens=sample.tokens,
                entities=entities
                ))

        predictions = self.__get_tags(predicted_data)
        labels = self.__get_tags(dataset_)
        predictions_concat = [j for sub in predictions for j in sub]
        labels_concat = [j for sub in labels for j in sub]

        # evaluate predictions
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


    def __average_score(self, score_, n_):
        assert isinstance(n_, int) and n_ > 0
        score_.precision /= n_
        score_.recall /= n_
        score_.f1 /= n_
        #score_.support /= n_
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
            else:
                res.accuracy += v.accuracy
                res.score_macro_avg = self.__accumulate_score(res.score_macro_avg, v.score_macro_avg)
                res.score_weighted_avg = self.__accumulate_score(res.score_weighted_avg, v.score_weighted_avg)
                res.score_entities = [self.__accumulate_score(ent_res, ent) for ent_res, ent in zip(res.score_entities, v.score_entities)]

            #self.__logger.info(f"Accumulation iter #{i+1}: {res}")

        # average scores
        assert self.__num_folds > 0
        res.accuracy /= self.__num_folds
        res.score_macro_avg = self.__average_score(res.score_macro_avg, self.__num_folds)
        res.score_weighted_avg = self.__average_score(res.score_weighted_avg, self.__num_folds)
        res.score_entities = [self.__average_score(ent_score, self.__num_folds) for ent_score in res.score_entities]

        return res

    def __get_tags(self, samples: List[Sample]) -> List[List[int]]:
        overall_tags = []
        for sample in samples:
            tags: List[int] = []
            for token in sample.tokens:
                found = False
                for entity in sample.entities:
                    if token.start >= entity.start and token.start < entity.end:
                        entity_type_as_integer = self.__entity_type_label_to_id_mapping[entity.ent_type.label] if entity.ent_type.label in self.__entity_type_label_to_id_mapping else self.__entity_type_set.id_of_non_entity
                        tags.append(entity_type_as_integer)
                        found = True
                        break
                if not found:
                    tags.append(self.__entity_type_set.id_of_non_entity)
            overall_tags.append(tags)
        return overall_tags


    def __convert_data_format(self, samples: list, filename: str):
        """ Split text and labels into tokens """
        doc_pipe = self.__spacy_model.pipe([sample.text for sample in samples])
        db = DocBin()
        for sample, doc in zip(samples, doc_pipe):
            ents = []
            for entity in sample.entities:
                if entity.ent_type.label in self.__entity_type_label_to_id_mapping:
                    span = doc.char_span(entity.start, entity.end, label=str(self.__entity_type_label_to_id_mapping[entity.ent_type.label]),
                            alignment_mode="expand")
                    ents.append(span)

            # filter overlapping entity spans (cf. https://spacy.io/api/top-level#util.filter_spans)
            ents_filtered = filter_spans(ents)
            doc.ents = ents_filtered
            db.add(doc)
        db.to_disk(self.__data_path + filename + ".spacy")
