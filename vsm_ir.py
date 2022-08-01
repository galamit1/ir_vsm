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
        self.max_appearance_per_file = {}
        self.squared_document_tf_idf_length = {}  # the sum of all the tf-idf ** 2 scores for each word in the file
        self.corpus = {}

        # nltk classes
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
                    if record_id not in self.squared_document_tf_idf_length:
                        self.squared_document_tf_idf_length[record_id] = 0

            self.words_per_file[record_id] = len(text) # TODO check if it's for the whole text
            text = self.tokenizer.tokenize(text.lower())  # tokens
            filtered_text = [self.ps.stem(word) for word in text if word not in self.stop_words]  # stopwords + stem

            self.update_dictionary(filtered_text, record_id)
            self.calculate_max_appearances(record_id)

    def update_dictionary(self, text, file_name):
        for word in text:
            if not self.dict_tf_idf_scores.get(word):
                self.dict_tf_idf_scores[word] = {}
                self.dict_tf_idf_scores[word][file_name] = {"count": 1, "tf_idf": 0}
            else:
                if self.dict_tf_idf_scores.get(word, {}).get(file_name):
                    self.dict_tf_idf_scores[word][file_name]["count"] += 1
                else:
                    self.dict_tf_idf_scores[word][file_name] = {"count": 1, "tf_idf": 0}

    def calculate_max_appearances(self, file_name):
        self.max_appearance_per_file[file_name] = 0
        for word_map in self.dict_tf_idf_scores.values():
            count = word_map.get(file_name, {}).get("count", 0)
            if count > self.max_appearance_per_file[file_name]:
                self.max_appearance_per_file[file_name] = count

    def calc_tf_idf_score(self, docs_count, avgdl):
        for word in self.dict_tf_idf_scores:
            for file in self.dict_tf_idf_scores[word]:
                word_frequency = self.dict_tf_idf_scores[word][file].get('count')

                # compute tf_idf
                tf = word_frequency / self.max_appearance_per_file.get(file) # TODO check if we want self.words_per_file.get(file)
                idf = np.log2(docs_count / len(self.dict_tf_idf_scores[word]))
                self.dict_tf_idf_scores[word][file]["tf_idf"] = tf * idf
                self.squared_document_tf_idf_length[file] += (tf * idf) ** 2

                # compute bm25
                bm25 = (idf * word_frequency * (self.K + 1)) / \
                       (word_frequency + self.K * (1 - self.B + self.B * self.words_per_file.get(file) / avgdl))
                self.dict_tf_idf_scores[word][file]["bm25"] = bm25

    def create_mapping(self, xml_dir_path):
        [self.parse_file(xml_dir_path + "/" + file_name) if file_name.endswith(".xml") else None for file_name in
         os.listdir(xml_dir_path)]

        amount_of_docs = len(self.squared_document_tf_idf_length)
        avgdl = sum(self.words_per_file.values()) / amount_of_docs
        self.calc_tf_idf_score(amount_of_docs, avgdl)

        # norm
        for file in self.squared_document_tf_idf_length:
            self.squared_document_tf_idf_length[file] = np.sqrt(self.squared_document_tf_idf_length[file])

        # add new dict to corpus
        corpus = {"dictionary": self.dict_tf_idf_scores,
                  "document_reference": self.squared_document_tf_idf_length}

        with open(INVERTED_INDEX_PATH, "w") as inverted_index_file:
            json.dump(corpus, inverted_index_file, indent=4)

    ### PART 2: Information Retrieval given a query. ###

    def load_ir(self, index_path):
        with open(index_path, "r") as inverted_index_file:
            corpus = json.load(inverted_index_file)

        self.dict_tf_idf_scores = corpus["dictionary"]
        self.squared_document_tf_idf_length = corpus["document_reference"]

    def normalize_query(self, query):
        query = self.tokenizer.tokenize(query.lower())  # tokens
        return [self.ps.stem(word) for word in query if word not in self.stop_words]  # stopwords + stem

    def perform_query(self, ranking_func, index_path, query): #TODO add bm25 support
        self.load_ir(index_path)
        query = self.normalize_query(query)
        query_tf_idf = self.calculate_query_tf_idf(query)

        relevant_docs = self.get_ranking(query_tf_idf)

        with open(OUTPUT_PATH, "w") as f:
            for i in range(0, len(relevant_docs)):
                if relevant_docs[i][1] >= 0.075:
                    f.write(relevant_docs[i][0] + "\n")

    # Create hashmap of dj * q for all documents that include words from query
    def get_documents_cossim_scores(self, query_map):
        documents_vectors = {}
        for word in query_map:
            if self.dict_tf_idf_scores.get(word):
                for doc in self.dict_tf_idf_scores[word]:
                    if doc not in documents_vectors:
                        documents_vectors[doc] = 0

                    documents_vectors[doc] += (self.dict_tf_idf_scores[word][doc]["tf_idf"] * query_map[word])

        return documents_vectors

    # Create sorted list of relevant documents by cosSim
    def get_ranking(self, query_map):
        results = []

        # Calc query vector length
        query_length = 0
        for token in query_map:
            query_length += (query_map[token] ** 2)
        query_length = np.sqrt(query_length)

        documents_scores = self.get_documents_cossim_scores(query_map)
        for doc in documents_scores:
            doc_query_product = documents_scores[doc]
            doc_length = np.sqrt(self.squared_document_tf_idf_length[doc])  # TODO we added the sqrt
            cosSim = doc_query_product / (doc_length * query_length)
            results.append((doc, cosSim))

        # Sort list by cosSim
        results.sort(key=lambda x: x[1], reverse=1)
        return results

    # Calculate query's tf-idf score.
    def calculate_query_tf_idf(self, query):
        number_of_docs = len(self.squared_document_tf_idf_length)
        query_tf_idf = {}
        max_word_count = max([query.count(word) for word in query])
        for word in set(query):
            tf = (query.count(word) / max_word_count) # TODO query_length = len(query)?
            n_word = len(self.dict_tf_idf_scores.get(word, {}))  # the number of documents with this word
            # TODO this is the BM25 idf
            bm_25_idf = np.log2(((number_of_docs - n_word + 0.5) / (n_word + 0.5)) + 1) \
                if self.dict_tf_idf_scores.get(word) else 0
            idf = np.log2(number_of_docs / len(self.dict_tf_idf_scores[word])) # number of docs / number of docs the word in
            query_tf_idf[str(word)] = tf * idf
        return query_tf_idf


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
    main()
