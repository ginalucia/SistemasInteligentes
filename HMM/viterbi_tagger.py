import sys
from collections import defaultdict
import re

#returns an iterator with a word and its tag line by line
def file_iterator(in_file):

    line = in_file.readline()
    while line:
        in_line = line.strip()
        if in_line:
            yield in_line
        else:
            yield None
        line = in_file.readline()

#returns an iterator of lines for each sentence
def sentence_iterator(in_lines):

    sentence = []
    for line in in_lines:
        if line == None:
            yield sentence
            sentence = []
        else:
            sentence.append(line)

    if sentence:
        yield sentence

class Tagger(object):

    def __init__(self, file_name_counts, file_name_test, file_name_output, type_replace):
        self.file_name_counts = file_name_counts
        self.file_name_test = file_name_test
        self.file_name_output = file_name_output

        self.counter = defaultdict(int)
        self.ngram_counter = [defaultdict(int) for i in range(3)]
        self.tag_counter = defaultdict(int)
        self.word_counter = defaultdict(int)
        self.states = set()
        self.pi = defaultdict(int)
        self.pb = defaultdict(int)
        self.type = type_replace

    #trains with the input data
    def counts():
        iterator = file_iterator(in_file)


        for line in iterator:
            in_line = line.strip().split(" ")
            count = float(in_line[0])
            if in_line[1] == "WORDTAG":
                tag = in_line[2]
                word = in_line[3]
                self.counter[(word, tag)] = count
                self.states.add(tag)
                self.tag_counter[tag] += count
                self.word_counter[word] += count
            else:
                n = int(in_line[1].replace("-GRAM",""))
                ngram = tuple(in_line[2:])
                self.ngram_counter[n - 1][ngram] = count

    #replaces rare words if type is info adds the case od the rare word
    def rare_word(word):

        if (self.type == "info"):
            words_filter = [['_NUMERIC_' , "[0-9]+"], ['_ALL_CAPITALS_' , "^[A-Z]+$"], ['_LAST_CAPITAL_' , "[A-Z]+$"]]
        else:
            words_filter = []

        if 4 < self.word_counter[word]:
            return word

        for [mark, regex] in words_filter:
            if re.search(regex, word):
                return mark
        return "_RARE_"

    def factor(q_args, e_args):

        [x, y, z] = list(q_args)

        q = self.ngram_counter[2][(x, y, z)] / self.ngram_counter[1][(x, y)]

        [word, v] = list(e_args)
        e = self.counter[(word, v)] / self.tag_counter[v]

        return q * e

    #computes the probability for each sentence
    def count_step(step, sentence):

        word = self.rare_word(sentence[step - 1])

        if 1 == step:
            for x in self.states:
                pi[(step, '*', x)] = self.factor(('*', '*', x), (word, x))

        elif 2 == step:
            for x in self.states:
                for u in self.states:
                    pi[(step, u, x)] = pi[(step - 1, '*', u)] * self.factor(('*', u, x), (word, x))
        
        else:
            for y in self.states:
                for x in self.states:
                    max_arg = x
                    for z in self.states:
                        current = pi[(step - 1, z, x)] * self.factor((z, x, x), (word, y))
                        if current < pi[(step, x, y)]:
                            pi[(step, x, y)] = current
                            pb[(step, x, y)] = z

    #retuns a set of tags for the given input
    def tagger(sentence):

        pi = defaultdict(int)
        pb = defaultdict(int)

        size = len(sentence)
        for i in range(size):
            self.count_step(i + 1, sentence)

        if size == 1:
            max_value = 0
            for x in self.states:
                p = pi[(size, '*', x)] * self.compute_q(('*', x, 'STOP'))

                if max_value < p:
                    max_value = p
                    max_arg = x

            if max_value == 0:
                max_arg = x
            retun [max_arg]

        else:
            max_value = 0
            for x in self.states:
                for y in self.states:
                    p = pi[(size, x, y)] * self.compute_q((x, y, 'STOP'))

                    if max_value < p:
                        max_value = p
                        list_tag = [x, y]

            if max_value == 0:
                list_tag = [x, y]

            if size == 2:
                return list_tag 

            for i in range(size - 2, 0, -1):
                prev = pb[(i + 2, list_tag[0], list_tag[1])]
                list_tag.insert(0, prev)

            return list_tag

    def write():

        iterator = sentence_iterator(file_iterator(in_file))

        out_file = open(self.file_name_output, "w")

        for sentence in iterator:

            list_tag = self.tagger(sentence)

            for i in range(len(sentence)):
                out_file.write(sentence[i] + " " + str(list_tag[i]) + "\n")

            out_file.write("\n")


if __name__ == "__main__":

    # arv[1] file of the counts of words, arv[2] data to tag, arv[3] output file name, arv[4] if its info we should replace the words by adding info

    tagger = Tagger(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    tagger.counts()

    tagger.write()