import sys
from collections import defaultdict
import math

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


class Tagger(object):

    def __init__(self, file_name_counts, file_name_test, file_name_output):
        self.file_name_counts = file_name_counts
        self.file_name_test = file_name_test
        self.file_name_output = file_name_output

        self.counter = defaultdict(int)
        self.tag_counter = defaultdict(int)
        self.states = set()


    def counts(self):

        line = file_iterator(open(self.file_name_input, "r"))

        for line in iterator:
            in_line = line.strip().split(" ")
            count = float(in_line[0])
            if in_line[1] == "WORDTAG":
                tag = in_line[2]
                word = in_line[3]
                self.counter[(word, tag)] = count
                self.states.add(tag)
                self.tag_counter[tag] += count
            else:
                n = int(in_line[1].replace("-GRAM",""))
                ngram = tuple(in_line[2:])


    def max_tag(self, word):
        count = 0
        new_counter = 0
        rare = True

        for tag in self.states:
            if 0 < self.counter[(word, tag)]:
                rare = False
                new_counter = self.counter[(word, tag)] / self.tag_counter[tag]
            if count < new_counter:
                count = new_counter
                new_tag = tag

        if rare == True:
            return max_tag("_RARE_")

        return new_tag


    def write(self):

        line = file_iterator(open(self.file_name_test, "r"))

        out_file = open(self.file_name_output, "w")

        for word in iterator:
            if word:
                out_file.write(word + " " + max_tag(word) + "\n")
            else:
                out_file.write("\n")


if __name__ == "__main__":

    # arv[1] file of the counts of words, arv[2] data to tag, arv[3] output file name

    tagger = Tagger(sys.argv[1], sys.argv[2], sys.argv[3])

    counts()

    write()
