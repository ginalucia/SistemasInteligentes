import sys
from collections import defaultdict


#returns an iterator with a word and its tag line by line
def file_iterator(in_file):

    l = in_file.readline()
    while l:
        line = l.strip()
        if line:
            fields = line.split(" ")
            tag = fields[-1]
            word = " ".join(fields[:-1])
            yield (word, tag)
        else:
            yield (None, None)
        l = in_file.readline()

class Replacer(object):

    def __init__(self, file_name_input, file_name_output, type_replace):
        self.counter = defaultdict(int)
        self.file_name_input = file_name_input
        self.file_name_output = file_name_output
        self.type = type_replace

    def word_counter(self):

        line = file_iterator(open(self.file_name_input, "r"))
        for word, tag in line:
            if word:
                self.counter[word] += 1

    def filter(self, word):

        if (self.type == "info"):
            words_filter = [['_NUMERIC_' , "[0-9]+"], ['_ALL_CAPITALS_' , "^[A-Z]+$"], ['_LAST_CAPITAL_' , "[A-Z]+$"]]
        else:
            words_filter = []

        for [info, regex] in words_filter:
            if re.search(regex, word):
                return info
        return '_RARE_'

    def replace(self):

        line = file_iterator(open(self.file_name_input, "r"))

        out_file = open(self.file_name_output, "w")

        for word, tag in line:
            if word is None:
                out_file.write("\n")
            else:
                if self.counter[word] < 5:
                    out_file.write("_RARE_ " + str(tag) + "\n")
                else:
                    out_file.write(word + " " + str( tag) + "\n")


if __name__ == "__main__":

    # arv[1] file of the counts of words, arv[2] output file name, arv[3] if its info we should replace the words by adding info

    replacer = Replacer(sys.argv[1], sys.argv[2], sys.argv[3])
            
    replacer.word_counter()
                        
    replacer.replace()