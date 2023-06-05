from typing import List

import spacy

from util.process_data import Token, Sample, SampleList

class Tokenizer():

    def __init__(self, spacy_model: str):
        self.__spacy_model = spacy.load(spacy_model)

    def run(self, sample_list: SampleList):
        self.__tokenize(sample_list.samples, self.__spacy_model)

    def __tokenize(self, samples: List[Sample], spacy_model):
        doc_pipe = spacy_model.pipe([sample.text.replace('\xa0', ' ') for sample in samples])
        for sample, doc in zip(samples, doc_pipe):
            sample.tokens = [Token(
                text=x.text,
                start=x.idx,
                end=x.idx + len(x.text)
            ) for x in doc]
            while '\n' in sample.tokens[-1].text or ' ' in sample.tokens[-1].text:
                sample.tokens = sample.tokens[:-1]
