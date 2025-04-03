import os
import math
from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# nltk.download()

class Document:
    def __init__(self, title, index, contents, file_path, file_name):
        self.title = title
        self.index = index
        self.contents = contents
        self.file_name = file_name
        self.file_path = file_path
        self.tokens = None
        self.tokens_rem_stopwords = None
        self.stemmed_tokens = None
        self.tf = None
        self.tf_idf_vector = None


class AllDocuments:
    def __init__(self, corpus_path):
        self.no_of_docs = 40
        self.path_to_docs = corpus_path
        self.documents = []
        self.df = defaultdict(int)
        self.idf = {}

    def read_all_docs(self):
        no_docs = 0
        for filename in os.listdir(self.path_to_docs):
            try:
                file_path = os.path.join(self.path_to_docs, filename)
                file = open(file_path, "r", encoding='windows-1252')
                content = file.read()
                content_lowered = content.lower()
                file.close()
                doc_obj = Document(title=filename[3:][:-4], index=int(filename[:2]), contents=content_lowered,
                                   file_path=file_path, file_name=filename)
                self.documents.append(doc_obj)
                no_docs += 1
            except ValueError:
                continue
        self.no_of_docs = no_docs

    def query_docs(self, index=None, title=None, file_path=None,
                   file_name=None):  # index=0, title="washington_1789", file_path=r"US_Inaugural_Addresses\01_washington_1789.txt"
        if index is not None:
            for each_doc in self.documents:
                if each_doc.index == index:
                    return each_doc
        if title is not None:
            for each_doc in self.documents:
                if each_doc.title == title:
                    return each_doc
        if file_path is not None:
            for each_doc in self.documents:
                if each_doc.file_path == file_path:
                    return each_doc
        if file_name is not None:
            for each_doc in self.documents:
                if each_doc.file_name == file_name:
                    return each_doc
        print("Query not found")
        return None

    def tokenize_docs(self):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+|\b\d{4}\b|\b\d{1,2}-\d{1,2}\b')  # to add dates as well
        for each_doc in self.documents:
            each_doc.tokens = tokenizer.tokenize(each_doc.contents)

    def remove_stopwords(self):
        stop_words = stopwords.words('english')
        for each_doc in self.documents:
            each_doc.tokens_rem_stopwords = [token for token in each_doc.tokens if token not in stop_words]

    def stem_docs(self):
        stemmer = PorterStemmer()
        for each_doc in self.documents:
            each_doc.stemmed_tokens = [stemmer.stem(token) for token in each_doc.tokens_rem_stopwords]

    def compute_tf_idf(self):
        for each_doc in self.documents:
            tf = defaultdict(int)
            for token in each_doc.stemmed_tokens:
                tf[token] += 1
            for token in set(each_doc.stemmed_tokens):  # unique only one max in each document
                self.df[token] += 1
            each_doc.tf = tf

        for token, df in self.df.items():
            self.idf[token] = math.log10(self.no_of_docs / df)

        for each_doc in self.documents:
            tf_idf_notNormalized = {}
            tf_idf_Normalized = {}
            for token, tf in each_doc.tf.items():
                if tf > 0:
                    tf_wt = 1 + math.log10(tf)
                else:
                    tf_wt = 0
                idf_wt = self.idf[token]
                tf_idf = tf_wt * idf_wt
                tf_idf_notNormalized[token] = tf_idf

            magnitude = math.sqrt(sum(each_tf_idf ** 2 for each_tf_idf in tf_idf_notNormalized.values()))
            if magnitude != 0:
                for token in tf_idf_notNormalized:
                    tf_idf_Normalized[token] = tf_idf_notNormalized[token] / magnitude
            else:
                tf_idf_Normalized = tf_idf_notNormalized  # just an edge case
            each_doc.tf_idf_vector = tf_idf_Normalized

    def instantiate(self):
        self.read_all_docs()
        self.tokenize_docs()
        self.remove_stopwords()
        self.stem_docs()
        self.compute_tf_idf()


class query_cl:
    def __init__(self, contents):
        self.contents = contents
        self.tokens = None
        self.tokens_rem_stopwords = None
        self.stemmed_tokens = None
        self.tf = None
        self.tf_idf_vector = None

    def convert_to_lower(self):
        self.contents = self.contents.lower()

    def tokenize_query(self):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+|\b\d{4}\b|\b\d{1,2}-\d{1,2}\b')
        self.tokens = tokenizer.tokenize(self.contents)

    def remove_stopwords(self):
        stop_words = stopwords.words('english')
        self.tokens_rem_stopwords = [token for token in self.tokens if token not in stop_words]

    def stem_query(self):
        stemmer = PorterStemmer()
        self.stemmed_tokens = [stemmer.stem(token) for token in self.tokens_rem_stopwords]

    def compute_tf(self):
        tf = defaultdict(int)
        for token in self.stemmed_tokens:
            tf[token] += 1
        self.tf = tf

        tf_notNormalized = {}
        tf_Normalized = {}
        for token, tf in self.tf.items():
            if tf > 0:
                tf_wt = 1 + math.log10(tf)
            else:
                tf_wt = 0
            idf_wt = 1  # no idf calculated for query
            tf_idf = tf_wt * idf_wt
            tf_notNormalized[token] = tf_idf

        magnitude = math.sqrt(sum(each_tf_idf ** 2 for each_tf_idf in tf_notNormalized.values()))
        if magnitude != 0:
            for token in tf_notNormalized:
                tf_Normalized[token] = tf_notNormalized[token] / magnitude
        else:
            tf_Normalized = tf_notNormalized  # just an edge case
        self.tf_idf_vector = tf_Normalized

    def instantiate(self):
        self.convert_to_lower()
        self.tokenize_query()
        self.remove_stopwords()
        self.stem_query()
        self.compute_tf()


class DataSearch:
    def __init__(self, documents_obj, query_obj):
        self.documents_obj = documents_obj
        self.documents = documents_obj.documents
        self.postings = defaultdict(list)
        self.query_obj = query_obj
        self.query = query_obj.contents
        self.top_10 = {}
        self.final_ranks = {}
        self.recommend = None

    def create_posting(self):
        for each_doc in self.documents:
            for token, wt in each_doc.tf_idf_vector.items():
                self.postings[token].append((each_doc.title, wt))
        for token in self.postings:
            self.postings[token].sort(key=lambda x: x[1], reverse=True)

    def calculate_top_10_docs(self):
        self.top_10 = {}
        for token in self.query_obj.stemmed_tokens:
            if token in self.postings:
                self.top_10[token] = self.postings[token][:10]

    def calculate_similarity(self, document_title):
        my_doc = self.documents_obj.query_docs(title=document_title)
        if my_doc is None:
            return 0

        sim = 0
        for token, query_wt in self.query_obj.tf_idf_vector.items():
            doc_wt = my_doc.tf_idf_vector.get(token, 0)
            sim += query_wt * doc_wt
        return sim

    def test_all_similarity(self):
        sets_of_titles = [set(title for title, tf_idf in query_token) for query_token in self.top_10.values()]
        if not sets_of_titles:
            self.final_ranks = {("None", 0)}  # Exception nothing matched
            self.recommend = ("None", 0)
            return
        common_titles = set.intersection(*sets_of_titles)
        self.final_ranks = {}
        if common_titles:  # case 7.5 all query top 10 document matches
            for title in common_titles:
                self.final_ranks[title] = self.calculate_similarity(title)
            self.final_ranks = dict(sorted(self.final_ranks.items(), key=lambda item: item[1], reverse=True))
            self.recommend = next(iter(self.final_ranks.items()))
        else:
            for token, query_wt in self.query_obj.tf_idf_vector.items():
                if token in self.top_10:  # nothing found for that token edge case
                    for doc_title, _ in self.top_10[token]:
                        if doc_title not in self.final_ranks:
                            self.final_ranks[doc_title] = (0, 0)

            for doc_title in self.final_ranks:
                actual_score = 0
                upper_bound = 0

                for token, query_wt in self.query_obj.tf_idf_vector.items():
                    if token in self.top_10:
                        top_10_titles = [title for title, _ in self.top_10[token]]
                        if doc_title in top_10_titles:
                            doc_wt = self.documents_obj.query_docs(title=doc_title).tf_idf_vector.get(token, 0)
                            actual_score += query_wt * doc_wt
                        else:
                            # Use the last (10th) element's weight as upper bound
                            try:
                                upper_bound_wt = self.top_10[token][-1][1]
                            except Exception as e:
                                print(e)
                                upper_bound_wt = 0  # edge case one of the token has no match
                            upper_bound += query_wt * upper_bound_wt
                self.final_ranks[doc_title] = (actual_score, actual_score + upper_bound)

            best_doc = None
            best_score = 0

            for doc_title, (actual_score, combined_score) in self.final_ranks.items():
                if actual_score >= best_score and all(
                        actual_score >= other_actual or actual_score >= other_upper
                        for other_title, (other_actual, other_upper) in self.final_ranks.items()
                        if other_title != doc_title
                ):
                    best_doc = (doc_title, actual_score)
                    best_score = actual_score

            if best_doc:
                self.recommend = best_doc
            else:
                self.recommend = ("fetch more", 0)
        return self.final_ranks

    def instantiate(self):
        self.create_posting()
        self.calculate_top_10_docs()
        self.test_all_similarity()


US_Inaugural_Addresses = AllDocuments(corpus_path='US_Inaugural_Addresses')
US_Inaugural_Addresses.instantiate()


def getidf(token):
    token = token.lower()
    stemmer = PorterStemmer()
    stemmed_token = stemmer.stem(token)
    try:
        return US_Inaugural_Addresses.idf[stemmed_token]
    except KeyError:
        return -1


def getweight(filename=None, token=None):
    token = token.lower()
    stemmer = PorterStemmer()
    stemmed_token = stemmer.stem(token)
    try:
        return US_Inaugural_Addresses.query_docs(file_name=filename).tf_idf_vector[stemmed_token]
    except KeyError:
        return 0


def query(qstring):
    myQuery = query_cl(qstring)
    myQuery.instantiate()
    mySearchEngine = DataSearch(US_Inaugural_Addresses, myQuery)
    mySearchEngine.instantiate()
    return mySearchEngine.recommend


# print(getidf("washington"))
# print(getweight(filename="01_washington_1789.txt", token="washington"))
# print(query("washington the volleyball genius damn this thing fetch more computers need more and more"))

print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt', 'constitution'))
print("%.12f" % getweight('23_hayes_1877.txt', 'public'))
print("%.12f" % getweight('25_cleveland_1885.txt', 'citizen'))
print("%.12f" % getweight('09_monroe_1821.txt', 'revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt', 'leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))
