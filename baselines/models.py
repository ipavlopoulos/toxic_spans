import pandas as pd
import random


class Random:

    def __init__(self, text):
        """
        Random baseline for toxic span detection.
        Returns toxic offsets at random.
        :param text:
        """
        self.toxic_offsets = [i for i, ch in enumerate(text) if random.random()>0.5]

    def get_toxic_offsets(self):
        return self.toxic_offsets


class InputErasure:

    def __init__(self,
                 classifier,
                 text,
                 one_by_one=False,
                 tokenise=lambda txt: txt.split(),
                 class_names=[0, 1],
                 mask=u"[mask]",
                 threshold=0.2,
                 reshape_predictions=True):
        """
        Given a classifier and a tokenisation method InputErasure returns the toxic words and respective offsets.
        This implementation is based on the paper "Understanding Neural Networks through Representation Erasure" by
        Li et al.

        :param classifier: any toxicity classifier that predicts a text as toxic or not
        :param text: the textual input (sentence or document) as a string
        :param one_by_one: some classifiers may require one by one classification when scoring the "ablated" texts.
        :param tokenise: by default splits the words on empty space
        :param class_names: by default "toxic" is represented by 1 and "civil" by 0
        :param mask: the pseudo token to mask the toxic word (for visualisation purposes)
        :param threshold: above this value the text is predicted toxic (default 0.2)
        :param reshape_predictions: flattens the output, some classifiers may required this to be set to False
        """
        self.class_names = class_names
        self.classifier = classifier
        self.mask = mask
        self.one_by_one = one_by_one
        self.reshape_predictions = reshape_predictions
        self.initial_score = self.clf_predict([text])
        self.tokenise = tokenise
        self.words = self.tokenise(text)
        self.ablations, self.indices = self.create_ablations()
        self.scores = self.clf_predict(self.ablations)
        self.e = 10e-05
        self.scores_decrease = [(self.initial_score - s) / (self.initial_score+self.e) for s in self.scores]
        self.threshold = threshold
        self.black_list = self.get_black_list()

    def clf_predict(self, texts):
        if self.one_by_one:
            predictions = [self.classifier.predict([t])[0] for t in texts]
        else:
            predictions = self.classifier.predict(texts)
        if self.reshape_predictions:
            predictions = predictions.reshape(1, -1)[0]
        if len(texts) == 1:
            return predictions[0]
        return predictions

    def create_ablations(self):
        ablations, indices = [], []
        for i, w in enumerate(self.words):
            words_copy = [w for w in self.words]
            words_copy[i] = self.mask
            ablations.append(" ".join(words_copy))
            indices.append(i)
        return ablations, indices

    def get_black_list(self):
        return [self.indices[i] for i, s in enumerate(self.scores_decrease) if s > self.threshold]

    def get_toxic_offsets(self):
        """
        Get the offsets of the toxic spans
        WARNING: Valid only for empty space tokenisation.
        :return: a list with offsets found toxic
        """
        current_offset = 0
        toxic_offsets = []
        for i, word in enumerate(self.words):
            if i in set(self.black_list):
                toxic_offsets.extend(list(range(current_offset, current_offset+len(word))))
            current_offset += len(word) + 1
        return toxic_offsets

    def get_mitigated_text(self):
        return " ".join([w if i not in set(self.black_list) else self.mask for i, w in enumerate(self.words)])

    def get_as_pandas(self):
        scores_pd = pd.DataFrame({"word": self.words, "indices": self.indices, "score_dec": self.scores_decrease})
        scores_pd = scores_pd.sort_values(by=["score_dec"])
        return scores_pd

