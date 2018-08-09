
from collections import Counter


class Vocabulary(object):

    def __init__(self, max_num_word):
        self.max_num_word = max_num_word
        self.oov_index = self.max_num_word + 1
        self.word_index = {}


    def fit(self, words_list):
        """
        :param words_list: [[w11, w12, ...], [w21, w22, ...], ...]
        :return:
        """
        word_lst = []
        word_lst_append = word_lst.append
        for words in words_list:
            if not isinstance(words, list):
                print(words)
                continue
            for word in words:
                word_lst_append(word)
        word_counts = Counter(word_lst)
        if self.max_num_word < 0:
            self.max_num_word = len(word_counts)
        sorted_voc = [w for w, c in word_counts.most_common(self.max_num_word)]
        self.max_num_word = len(sorted_voc)
        self.oov_index = self.max_num_word + 1
        self.word_index = dict(zip(sorted_voc, range(1, self.max_num_word+1)))
        return self


    def _transform_inner(self, words):
        vect = []
        vect_append = vect.append
        for w in words:
            if w in self.word_index:
                vect_append(self.word_index[w])
            else:
                vect_append(self.oov_index)
        return vect


    def transform(self, words_list):
        return [self._transform_inner(words) for words in words_list]


    def fit_transform(self, words_list):
        self.fit(words_list)
        return self.transform(words_list)