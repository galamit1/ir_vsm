# Information Retrieval - Vector Space Model
The last project in Web Data Management course at Tel Aviv University.


## Part 1 - Build an inverted index from a given documents.
`python vsm_ir.py create_index cfc-xml`

In this part, the program extracts the data from the documents, performs tokenizing stemming, and removes the stops words.
Afterward, the program calculates more data for the inverted index, for example, the tf-idf scores for each word for every document, and saves it in the file "vsm_inverted_index.json".

## Part 2 - Retrieve documents from a query
`python vsm_ir.py query [ranking] [index_path] “<question>”`

The program calculates the ranking for each document that can be "bm25" or "tf-idf".

In the end, the program saves in "ranked_query_docs.txt" the names of the documents with the highest ranking.
