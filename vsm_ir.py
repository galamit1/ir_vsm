import nltk

INVERTED_INDEX_PATH = "vsm_inverted_index.json"

class InvertedIndex(object):
    def __init__(self):
        self.tf = {}
        self.idf = {}
        self.document_length = {}



def parse_text(txt):
    words = nltk.tokenize.word_tokenize(txt, preserve_line=False)  # tokenize
    stemmer = nltk.PorterStemmer()
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]  # removing stopwords and stemming



def create_index(xml_dir):
    return


def main():
    create_index("blash")


if __name__ == '__main__':
    main()
