from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import numpy as np
import nltk
import json
import sys
import os
import ssl

INVERTED_INDEX_PATH = "vsm_inverted_index.json"
OUTPUT_PATH = "ranked_query_docs.txt"


class IR(object):
    # bm25 constants
    K = 1.2
    B = 0.75

    def __init__(self):
        # inverted index
        self.dict_tf_idf_scores = {}
        self.words_per_file = {}
        self.query_dictionary = {}
        self.document_reference_length = {}
        self.corpus = {}

        # nltk classes
        try:
            nltk.download('stopwords')
        except:  # disable SSL check. reference: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ps = PorterStemmer()

    # part 1: calculate tf-idf scores

    def parse_file(self, filename):
        xml_tree = ET.parse(filename)
        root = xml_tree.getroot()

        for child in root.findall("./RECORD"):
            record_id = 0
            text = ""
            for entry in child:
                if entry.tag == "TITLE" or entry.tag == "EXTRACT" or entry.tag == "ABSTRACT":
                    text += str(entry.text) + " "
                if entry.tag == "RECORDNUM":
                    record_id = int(entry.text)
                    if record_id not in self.document_reference_length:
                        self.document_reference_length[record_id] = 0

            text = self.tokenizer.tokenize(text.lower())  # tokens
            filtered_text = [self.ps.stem(word) for word in text if word not in self.stop_words]  # stopwords + stem

            self.update_dictionary(filtered_text, record_id)
            self.words_per_file[record_id] = len(text)

    def update_dictionary(self, text, file_name):
        for word in text:
            if self.dict_tf_idf_scores.get(word, {}).get(file_name):
                self.dict_tf_idf_scores[word][file_name]["count"] += 1
            else:
                self.dict_tf_idf_scores[word] = {file_name: {"count": 1, "tf_idf": 0}}

    def calc_tf_idf_score(self, docs_count, avgdl):
        for word in self.dict_tf_idf_scores:
            for file in self.dict_tf_idf_scores[word]:
                word_frequency = self.dict_tf_idf_scores[word][file].get('count')

                # compute tf_idf
                tf = word_frequency / self.words_per_file.get(file)
                idf = np.log2(docs_count / len(self.dict_tf_idf_scores[word]))
                self.dict_tf_idf_scores[word][file]["tf_idf"] = tf * idf
                self.document_reference_length[file] += (tf * idf) ** 2

                # compute bm25
                bm25 = (idf * word_frequency * (self.K + 1)) / \
                       (word_frequency + self.K * (1 - self.B + self.B * self.words_per_file.get(file) / avgdl))
                self.dict_tf_idf_scores[word][file]["bm25"] = bm25

    def create_mapping(self, xml_dir_path):
        [self.parse_file(xml_dir_path + "/" + file_name) if file_name.endswith(".xml") else None for file_name in
         os.listdir(xml_dir_path)]

        amount_of_docs = len(self.document_reference_length)
        avgdl = sum(self.words_per_file.values()) / amount_of_docs
        self.calc_tf_idf_score(amount_of_docs, avgdl)

        # norm
        for file in self.document_reference_length:
            self.document_reference_length[file] = np.sqrt(self.document_reference_length[file])

        # add new dict to corpus
        corpus = {"dictionary": self.dict_tf_idf_scores,
                  "document_reference": self.document_reference_length}

        with open(INVERTED_INDEX_PATH, "w") as inverted_index_file:
            json.dump(corpus, inverted_index_file)

    ### PART 2: Information Retrieval given a query. ###

    def load_ir(self, index_path):
        with open(index_path, "r") as inverted_index_file:
            corpus = json.load(inverted_index_file)

        self.dict_tf_idf_scores = corpus["dictionary"]
        self.document_reference_length = corpus["document_reference"]

    def normalize_query(self, query):
        query = self.tokenizer.tokenize(query.lower())  # tokens
        return [self.ps.stem(word) for word in query if word not in self.stop_words]  # stopwords + stem

    def perform_query(self, ranking_func, index_path, query):
        self.load_ir(index_path)
        query = self.normalize_query(query)


def main():
    # call methods based on system arguments
    ir = IR()
    try:
        if sys.argv[1] == 'create_index':
            return ir.create_mapping(sys.argv[2])
        elif sys.argv[1] == 'query':
            return ir.perform_query(*sys.argv[2:5])
    except IndexError:
        print("Wrong Input")


if __name__ == '__main__':
    main()
