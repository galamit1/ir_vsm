import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET
import numpy as np
import json
import sys
import os
import ssl
from collections import defaultdict

INVERTED_INDEX_PATH = "vsm_inverted_index.json"
OUTPUT_PATH = "ranked_query_docs.txt"

COUNT = "count"
TFIDF = "tfidf"
BM25 = "bm25"


class IR(object):
    # bm25 constants
    K = 1.2
    B = 0.75

    def __init__(self):
        # inverted index
        self.dict_tf_idf_scores = {}
        self.words_per_file = defaultdict(int)
        self.max_appearance_per_file = defaultdict(int)
        # the sum of all the tf-idf ** 2 scores for each word in the file,
        # its on purpose not a default dict because we are using it to get the number of documents
        self.squared_document_tf_idf_length = {}

        # nltk classes
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ps = PorterStemmer()

    ### PART 1: Calculate tf-idf scores ###

    def parse_file(self, filename):
        """
        extract the relevant words from the file, perform tokenizing, stemming and remove stop words.
        save the data into the relevant dicts that will be saved in the inverted index.
        """
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

            self.words_per_file[record_id] = len(text)
            text = self.tokenizer.tokenize(text.lower())  # tokens
            filtered_text = [self.ps.stem(word) for word in text if word not in self.stop_words]  # stopwords + stem

            self.update_dictionary_count(filtered_text, record_id)
            self.calculate_max_appearances(record_id)

    def update_dictionary_count(self, text, file_name):
        """
        update the count field in the inverted index.
        at the end we will have for each word - number of appearance in every file.
        """
        for word in text:
            if not self.dict_tf_idf_scores.get(word):
                self.dict_tf_idf_scores[word] = {}
                self.dict_tf_idf_scores[word][file_name] = {COUNT: 1}
            else:
                if self.dict_tf_idf_scores[word].get(file_name):
                    self.dict_tf_idf_scores[word][file_name][COUNT] += 1
                else:
                    self.dict_tf_idf_scores[word][file_name] = {COUNT: 1}

    def calculate_max_appearances(self, file_name):
        """
        create a dictionary that will be saved in the inverted index that contains for each file, the max appearance number for word.
        it will be used in the calculation of tfidf scores.
        """
        for word_map in self.dict_tf_idf_scores.values():
            count = word_map.get(file_name, {}).get(COUNT, 0)
            if count > self.max_appearance_per_file[file_name]:
                self.max_appearance_per_file[file_name] = count

    def calc_tf_idf_score(self):
        """
        calculate the tfidf score for each word, for every file.
        we will use it in the cossim calculation when performing the query.
        """
        docs_number = len(self.squared_document_tf_idf_length)
        for word in self.dict_tf_idf_scores:
            for file in self.dict_tf_idf_scores[word]:
                word_frequency = self.dict_tf_idf_scores[word][file].get(COUNT)

                # compute tf_idf
                tf = word_frequency / self.max_appearance_per_file[file]
                idf = np.log2(docs_number / len(self.dict_tf_idf_scores[word]))
                self.dict_tf_idf_scores[word][file][TFIDF] = tf * idf
                self.squared_document_tf_idf_length[file] += (tf * idf) ** 2

    def create_mapping(self, xml_dir_path):
        """
        the main function of creating the index that calculates the relevant dictionaries and save it in the inverted index path.
        """
        [self.parse_file(xml_dir_path + "/" + file_name) if file_name.endswith(".xml") else None for file_name in
         os.listdir(xml_dir_path)]

        self.calc_tf_idf_score()

        # calculate squared document tfidf length for cossim normalization
        for file in self.squared_document_tf_idf_length:
            self.squared_document_tf_idf_length[file] = np.sqrt(self.squared_document_tf_idf_length[file])

        # add new dict to corpus
        corpus = {"dictionary": self.dict_tf_idf_scores,
                  "squared_document_tf_idf_length": self.squared_document_tf_idf_length,
                  "words_per_file": self.words_per_file,
                  "max_appearance_per_file": self.max_appearance_per_file}

        with open(INVERTED_INDEX_PATH, "w") as inverted_index_file:
            json.dump(corpus, inverted_index_file, indent=4)

    ### PART 2: Information Retrieval given a query. ###

    def load_ir(self, index_path):
        """
        loads the dictionaries from the inverted index in the given path.
        """
        with open(index_path, "r") as inverted_index_file:
            corpus = json.load(inverted_index_file)

        self.dict_tf_idf_scores = corpus["dictionary"]
        self.squared_document_tf_idf_length = corpus["squared_document_tf_idf_length"]
        self.words_per_file = corpus["words_per_file"]
        self.max_appearance_per_file = corpus["max_appearance_per_file"]

    def normalize_query(self, query):
        """
        normalize the given query by tokenizing, stemming and stop words removal.
        """
        query = self.tokenizer.tokenize(query.lower())  # tokens
        return [self.ps.stem(word) for word in query if word not in self.stop_words]  # stopwords + stem

    def get_documents_cossim_scores(self, query_map):
        """
        perform the cossim calculation, return a dict of dj * q for all documents that include words from query.
        """
        documents_vectors = {}
        for word in query_map:
            if self.dict_tf_idf_scores.get(word):
                for doc in self.dict_tf_idf_scores[word]:
                    if doc not in documents_vectors:
                        documents_vectors[doc] = 0
                    documents_vectors[doc] += (self.dict_tf_idf_scores[word][doc][TFIDF] * query_map[word])

        return documents_vectors

    def get_tfidf_ranking(self, query):
        """
        calculate the query tfidf and create sorted list of relevant documents by the cosSim score.
        """
        query_map = self.calculate_query_tf_idf(query)
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

    def calculate_query_tf_idf(self, query):
        """
        calculate the tfidf score for each word in the query.
        """
        number_of_docs = len(self.squared_document_tf_idf_length)
        query_tf_idf = defaultdict(int)
        max_word_count = max([query.count(word) for word in query])
        for word in set(query):
            if word not in self.dict_tf_idf_scores:
                continue
            tf = (query.count(word) / max_word_count)
            idf = np.log2(
                number_of_docs / len(self.dict_tf_idf_scores.get(word)))  # number of docs / number of docs the word in
            query_tf_idf[str(word)] = tf * idf
        return query_tf_idf

    # Calculate query's bm25 score.
    def get_bm25_ranking(self, query):
        """
        calculate the documents bm25 scores using the query and create sorted list of the documents by the scores.
        """
        number_of_docs = len(self.squared_document_tf_idf_length)
        avgdl = sum(self.words_per_file.values()) / number_of_docs

        documents_scores = defaultdict(int)
        for word in query:
            n_word = len(self.dict_tf_idf_scores.get(word, {}))  # the number of documents with this word
            bm25_idf = np.log2(((number_of_docs - n_word + 0.5) / (n_word + 0.5)) + 1) \
                if self.dict_tf_idf_scores.get(word) else 0

            for doc in self.dict_tf_idf_scores.get(word, {}).keys():
                word_frequency = self.dict_tf_idf_scores[word][doc][COUNT]
                bm25_score_for_word = (bm25_idf * word_frequency * (self.K + 1)) / \
                                      (word_frequency + self.K * (
                                              1 - self.B + self.B * self.words_per_file[doc] / avgdl))
                documents_scores[doc] += bm25_score_for_word

        results = list(documents_scores.items())
        results.sort(key=lambda x: x[1], reverse=1)
        return results

    def perform_query(self, ranking_func, index_path, query):
        """
        write to the output path the results of the query with the given ranking function.
        """
        self.load_ir(index_path)
        query = self.normalize_query(query)
        if ranking_func == TFIDF:
            relevant_docs = self.get_tfidf_ranking(query)
        elif ranking_func == BM25:
            relevant_docs = self.get_bm25_ranking(query)
        else:
            raise ("Invalid ranking function: " + ranking_func)

        with open(OUTPUT_PATH, "w") as f:
            for i in range(0, len(relevant_docs)):
                if relevant_docs[i][1] >= 0.075:
                    f.write(relevant_docs[i][0] + "\n")


def main():
    # call methods based on the system arguments
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
        nltk.download('popular', quiet=True)
    except:  # disable SSL check. reference: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('popular', quiet=True)
    main()
