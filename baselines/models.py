import pandas as pd


class InputErasure:

    def __init__(self, classifier, text, one_by_one=False, class_names=[0, 1], mask=u"[mask]", threshold=0.2, reshape_predictions=True):
        self.class_names = class_names
        self.classifier = classifier
        self.mask = mask
        self.one_by_one = one_by_one
        self.reshape_predictions = reshape_predictions
        self.initial_score = self.clf_predict([text])
        self.words = self.classifier.tokenise(text)
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

    def get_mitigated_text(self):
        return " ".join([w if i not in set(self.black_list) else self.mask for i, w in enumerate(self.words)])

    def get_as_pandas(self):
        scores_pd = pd.DataFrame({"word": self.words, "indices": self.indices, "score_dec": self.scores_decrease})
        scores_pd = scores_pd.sort_values(by=["score_dec"])
        return scores_pd

