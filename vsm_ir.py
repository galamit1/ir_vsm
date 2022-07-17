from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import numpy as np
import nltk
import json
import sys
import os

INVERTED_INDEX_PATH = "vsm_inverted_index.json"

# This dictionary holds all the words, the tf-idf for each file for every word.
dict_tf_idf_scores = {}
words_per_file = {}
query_dictionary = {}
document_reference_length = {}
corpus = {}


# part 1: calculate tf-idf scores


def parse_file(filename):
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


def calc_tf_idf_score(docs_count):
    for word in dict_tf_idf_scores:
        for file in dict_tf_idf_scores[word]:
            tf = dict_tf_idf_scores[word][file].get('count') / words_per_file.get(file)
            idf = np.log2(docs_count / len(dict_tf_idf_scores[word]))
            dict_tf_idf_scores[word][file]["tf_idf"] = tf * idf
            document_reference_length[file] += (tf * idf) ** 2


def create_mapping():
    [parse_file(sys.argv[2] + "/" + file_name) if file_name.endswith(".xml") else None for file_name in
     os.listdir(sys.argv[2])]

    amount_of_docs = len(document_reference_length)
    calc_tf_idf_score(amount_of_docs)

    # norm
    for file in document_reference_length:
        document_reference_length[file] = np.sqrt(document_reference_length[file])

    # add new dict to corpus
    corpus["dictionary"] = dict_tf_idf_scores
    corpus["document_reference"] = document_reference_length

    with open("vsm_inverted_index.json", "w") as inverted_index_file:
        json.dump(corpus, inverted_index_file, indent=8)


### PART 2: Information Retrieval given a query. ###


def query():
    pass


def main():
    # call methods based on system arguments
    create_mapping() if sys.argv[1] == 'create_index' else (
        query() if sys.argv[1] == 'query' else "Illegal Input! \n please insert correct instruction  :)")


if __name__ == '__main__':
    main()
