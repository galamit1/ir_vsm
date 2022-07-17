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

# bm25 constants
K = 1.2
B = 0.75

# This dictionary holds all the words, the tf-idf for each file for every word.
dict_tf_idf_scores = {}
words_per_file = {}
query_dictionary = {}
document_reference_length = {}
corpus = {}

# part 1: calculate tf-idf scores


def parse_file(filename):
    try: #TODO remove this block
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
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
                if record_id not in document_reference_length:
                    document_reference_length[record_id] = 0

        text = tokenizer.tokenize(text.lower())  # tokens
        filtered_text = [ps.stem(word) for word in text if word not in stop_words]  # stopwords + stem

        update_dictionary(filtered_text, record_id)
        words_per_file[record_id] = len(text)


def update_dictionary(text, file_name):
    for word in text:
        try:
            if dict_tf_idf_scores.get(word).get(file_name):
                dict_tf_idf_scores[word][file_name]["count"] += 1
        except:
            dict_tf_idf_scores[word] = {file_name: {"count": 1, "tf_idf": 0}}


def calc_tf_idf_score(docs_count, avgdl):
    for word in dict_tf_idf_scores:
        for file in dict_tf_idf_scores[word]:
            word_frequency = dict_tf_idf_scores[word][file].get('count')

            # compute tf_idf
            tf = word_frequency / words_per_file.get(file)
            idf = np.log2(docs_count / len(dict_tf_idf_scores[word]))
            dict_tf_idf_scores[word][file]["tf_idf"] = tf * idf
            document_reference_length[file] += (tf * idf) ** 2

            # compute bm25
            bm25 = (idf * word_frequency * (K + 1)) / \
                   (word_frequency + K * (1 - B + B * words_per_file.get(file) / avgdl))
            dict_tf_idf_scores[word][file]["bm25"] = bm25


def create_mapping(xml_dir_path):
    [parse_file(xml_dir_path + "/" + file_name) if file_name.endswith(".xml") else None for file_name in
     os.listdir(xml_dir_path)]

    amount_of_docs = len(document_reference_length)
    avgdl = sum(words_per_file.values()) / amount_of_docs
    calc_tf_idf_score(amount_of_docs, avgdl)

    # norm
    for file in document_reference_length:
        document_reference_length[file] = np.sqrt(document_reference_length[file])

    # add new dict to corpus
    corpus["dictionary"] = dict_tf_idf_scores
    corpus["document_reference"] = document_reference_length

    with open(INVERTED_INDEX_PATH, "w") as inverted_index_file:
        json.dump(corpus, inverted_index_file)


### PART 2: Information Retrieval given a query. ###

def query(ranking_func, index_path, question):
    with open(index_path, "r") as inverted_index_file:
        corpus = json.load(inverted_index_file)

    dict_tf_idf_scores.update(corpus["dictionary"])
    document_reference_length.update(corpus["document_reference"])


def main():
    # call methods based on system arguments
    try:
        if sys.argv[1] == 'create_index':
            return create_mapping(sys.argv[2])
        elif sys.argv[1] == 'query':
            return query(*sys.argv[2:5])
    except ValueError:
        print("Wrong Input")



if __name__ == '__main__':
    main()
