import math
from collections import defaultdict, Counter

class MarkovLangModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.counts = defaultdict(Counter)   # prev_char -> Counter(next_char)
        self.vocab = set()
        self.trained = False

    def fit(self, sentences):
        BOS = "<s>"; EOS = "</s>"
        self.vocab.update([BOS, EOS])
        for s in sentences:
            seq = [BOS] + list(s) + [EOS]
            for a, b in zip(seq[:-1], seq[1:]):
                self.counts[a][b] += 1
                self.vocab.update([a,b])
        self.trained = True

    def logprob(self, sentence):
        assert self.trained
        BOS = "<s>"; EOS = "</s>"
        V = len(self.vocab)
        lp = 0.0
        for a, b in zip([BOS]+list(sentence), list(sentence)+[EOS]):
            num = self.counts[a][b] + self.alpha
            den = sum(self.counts[a].values()) + self.alpha * V
            lp += math.log(num) - math.log(den)
        return lp

class MarkovClassifier:
    def __init__(self, alpha=0.5):
        self.models = {}
        self.alpha = alpha

    def fit_per_language(self, data_by_lang):
        for lang, sents in data_by_lang.items():
            m = MarkovLangModel(alpha=self.alpha)
            m.fit(sents)
            self.models[lang] = m

    def score(self, sentence):
        # returns {lang: logprob}
        return {lang: m.logprob(sentence) for lang, m in self.models.items()}

    def predict(self, sentence):
        scores = self.score(sentence)
        return max(scores, key=scores.get), scores