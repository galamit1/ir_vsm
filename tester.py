import math
import xml.etree.ElementTree as ET

import os

################ CHANGE IT IF NEEDED #################
CORPUS_DIRECTORY = 'cfc-xml_corrected'
EXAMPLE_QUERIES_PATH = 'cfc-xml_corrected/cfquery.xml'
INDEX_PATH = 'vsm_inverted_index.json'
RESULTS_PATH = 'ranked_query_docs.txt'
######################################################

TFIDF = 'tfidf'
BM25 = 'bm25'

# IDCG10 = 8 + sum([8/math.log2(i) for i in range(2, 11)])

last_modify_time = 0  # Last time of reading ranked_query_docs.txt file. Used to prevent reading an old results file


def gain(gain_str: str) -> int:
    gain_sum = 0
    for char in gain_str:
        gain_sum += int(char)
    return gain_sum


def idcg10(relevant_documents: dict) -> float:
    sorted_relevant_documents = sorted(relevant_documents, key=lambda x: relevant_documents[x], reverse=True)
    n = len(sorted_relevant_documents)
    idcg = 0
    if n > 0:
        idcg += relevant_documents[sorted_relevant_documents[0]]
        for i in range(1, min(n, 10)):
            idcg += relevant_documents[sorted_relevant_documents[i]] / math.log2(i+1)
    return idcg


def test_results(query_element: ET.Element) -> tuple:
    global last_modify_time
    new_modify_time = os.path.getmtime(RESULTS_PATH)
    while new_modify_time == last_modify_time:
        new_modify_time = os.path.getmtime(RESULTS_PATH)
    last_modify_time = new_modify_time

    with open(RESULTS_PATH, 'r') as file:
        lines = file.read().splitlines()
        retrieved_documents = [int(line.rstrip()) for line in lines]

    n = len(retrieved_documents)
    if n == 0:
        return 0, 0, 0, 0

    relevant_documents = {int(item.text): gain(item.get('score')) for item in query_element.findall('Records/Item')}
    num_of_relevant_documents = int(query_element.find("Results").text)
    num_of_retrieved_documents = len(retrieved_documents)
    intersection_size = len(set(relevant_documents) & set(retrieved_documents))

    precision = intersection_size / num_of_retrieved_documents
    recall = intersection_size / num_of_relevant_documents
    try:
        f = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f = 0

    dcg10 = 0.0
    if retrieved_documents[0] in relevant_documents:
        dcg10 += relevant_documents[retrieved_documents[0]]
    for i in range(1, 10):
        if i < n and retrieved_documents[i] in relevant_documents:
            dcg10 += relevant_documents[retrieved_documents[i]]/math.log2(i+1)  # retrieved_documents[i] is the (i+1)th result
    ndcg10 = dcg10 / idcg10(relevant_documents)

    return ndcg10, precision, recall, f


if __name__ == '__main__':
    os.system(f'python vsm_ir.py create_index {CORPUS_DIRECTORY}')
    print('Finished creating index')
    print()

    root = ET.parse(EXAMPLE_QUERIES_PATH).getroot()
    queries = root.findall('QUERY')
    count = 0

    avg_ndcg10_tf_idf = 0
    avg_precision_tf_idf = 0
    avg_recall_tf_idf = 0
    avg_f_tf_idf = 0

    avg_ndcg10_bm25 = 0
    avg_precision_bm25 = 0
    avg_recall_bm25 = 0
    avg_f_bm25 = 0

    for query in queries:
        count += 1
        query_text = query.find('QueryText').text.replace('\n', ' ')

        print(f'Query #{count}:')

        os.system(f'python vsm_ir.py query {TFIDF} {INDEX_PATH} "{query_text}"')
        _ndcg10, _precision, _recall, _f = test_results(query)
        avg_ndcg10_tf_idf += _ndcg10
        avg_precision_tf_idf += _precision
        avg_recall_tf_idf += _recall
        avg_f_tf_idf += _f
        print(f'TF-IDF Scores:\nNDCG@10: {_ndcg10},\t\tPrecision: {_precision},\t\tRecall: {_recall},\t\tF: {_f}')

        os.system(f'python vsm_ir.py query {BM25} {INDEX_PATH} "{query_text}"')
        _ndcg10, _precision, _recall, _f = test_results(query)
        avg_ndcg10_bm25 += _ndcg10
        avg_precision_bm25 += _precision
        avg_recall_bm25 += _recall
        avg_f_bm25 += _f
        print(f'BM25 Scores:\nNDCG@10: {_ndcg10},\t\tPrecision: {_precision},\t\tRecall: {_recall},\t\tF: {_f}')
        print()

    print()

    avg_ndcg10_tf_idf /= count
    avg_precision_tf_idf /= count
    avg_recall_tf_idf /= count
    avg_f_tf_idf /= count
    print('AVERAGE:')
    print(f'TF-IDF Scores:\nNDCG@10: {avg_ndcg10_tf_idf},\t\tPrecision: {avg_precision_tf_idf},\t\tRecall: {avg_recall_tf_idf},\t\tF: {avg_f_tf_idf}')

    avg_ndcg10_bm25 /= count
    avg_precision_bm25 /= count
    avg_recall_bm25 /= count
    avg_f_bm25 /= count
    print(f'BM25 Scores:\nNDCG@10: {avg_ndcg10_bm25},\t\tPrecision: {avg_precision_bm25},\t\tRecall: {avg_recall_bm25},\t\tF: {avg_f_bm25}')
