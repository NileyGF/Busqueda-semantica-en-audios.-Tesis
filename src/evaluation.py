from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
# from shared.utils import load_from_json
# import textdistance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import src.embedding_retrieval as emb_ret
from src.core import directory_path

def hits(qrels: list, retrieved: list, top_k:int = 0):
    """ Number of retrieved relevant songs. Returns how many elements of qrels are in retrieved, using numpy 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        top_k:       if top_k > 0, only the top_k retrieved songs are considered
    """
    k = top_k if top_k > 0 else len(retrieved)
    considered = retrieved[:k]
    intersect = np.intersect1d(qrels, considered)
    # print(intersect)
    return len(intersect)

def precision(qrels: list, retrieved: list, top_k:int = 0):
    """ Proportion of the retrieved songs that are relevant. 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        top_k:       if top_k > 0 only the top_k retrieved songs are considered
        
        Precision = r / n
        r: number of retrieved relevant songs
        n: number of retrieved songs

        if top_k > 0:
            Precision@k = r_k / top_k
            r_k:  number of retrieved relevant songs at top_k
    """
    if len(retrieved) == 0: return 0
    k = top_k if top_k > 0 else len(retrieved)
    retrieved_relevants = hits(qrels, retrieved, k)
    return retrieved_relevants  / k

def recall(qrels: list, retrieved: list, top_k:int):
    """ Ratio between the retrieved songs that are relevant and the total number of relevant songs. 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        top_k:       if top_k > 0 only the top_k retrieved songs are considered
        
        Recall = r/R
        r: number of retrieved relevant songs
        R: total number of relevant songs

        if top_k > 0:
            Recall@k = r_k / R
            r_k:  number of retrieved relevant songs at top_k
            R:    total number of relevant songs
    """
    if len(qrels) == 0: return 0
    k = top_k if top_k > 0 else len(retrieved)
    retrieved_relevants = hits(qrels, retrieved, k)
    return retrieved_relevants / len(qrels)

def f_metric(qrels: list, retrieved: list, beta = 1):
    """ Weighted harmonic mean of Precision and Recall. 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        F = ((1+beta^2) * P * R ) / ( beta^2 * P + R )
        beta: weight
        P: precision
        R: recall
    """    
    if beta <= 0: return None
    precision_s = precision(qrels, retrieved)
    recall_s = recall(qrels, retrieved)
    if precision_s == 0 or recall_s == 0: return 0
    return ((1 + beta ** 2) * precision_s * recall_s) / ((beta ** 2) * precision_s + recall_s)

def precision_ranked(qrels: list, retrieved: list, top_k:int):
    """ Proportion of the retrieved songs that are relevant. 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        R-Precision = r/R
        r: number of relevant songs among the top-R retrieved
        R: total number of relevant songs
        R-Precision is equal to recall at the R-th position
    """
    # R = min(len(qrels), len(retrieved))
    hits_on_top_R = hits(qrels, retrieved[:top_k])
    if len(qrels) == 0: return 0
    return hits_on_top_R / len(qrels)

def fallout(qrels: list, retrieved: list, total_docs: int):
    """ Proportion of non-relevant songs retrieved,
        out of all non-relevant songs available 
        fallout = nr/nn
        nr: number of non-relevant songs retrieved
        nn: total of non-relevant songs
    """
    non_hits = len(retrieved) - hits(qrels, retrieved)
    non_rel_docs = total_docs - len(qrels)
    if non_hits == 0 or non_rel_docs == 0: return 0
    return non_hits / non_rel_docs

def plot(X:list, Y:list, avg: float, name:str, fig: int, Xlabel = 'queries', Ylabel=''):
    # path = os.path.join(os.path.dirname(__file__), 'evaluation')
    # path = os.path.join(path, name)
    plt.figure(fig)
    plt.scatter(X, Y )
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title('Average '+name+': '+str(round(avg,4)))
    # plt.savefig(path, format='png')
    plt.show()
    
def evaluate_full_dataset(corpus_embedd:str, queries_embedd:str, relevance:str, F_beta: float, top_k:int):
    with open(corpus_embedd, 'rb') as f:
        corpus_embeddings_list = pickle.load(f)
    with open(queries_embedd, 'rb') as f:
        queries_embeddings_list = pickle.load(f) 
    with open(relevance, 'rb') as f:
        relevance_sim_list = pickle.load(f) 
    relevance_list = []
    for l in relevance_sim_list:
        l1 = [i for i,j in l]
        relevance_list.append(l1)
    # print(relevance_list[:3])

    st = time.time()
    
    precision_50_l = []
    precision_20_l = []
    precision_10_l = []
    precision_1_l = []
    recall_50_l = []
    recall_20_l = []
    recall_10_l = []
    recall_1_l = []
    for q in range(len(queries_embeddings_list)):
        retrieved =  emb_ret.evaluate_query(query_vector=queries_embeddings_list[q], documents_vectors_list=corpus_embeddings_list, top_k='all')
        qrels = relevance_list[q]
        # if q not in retrieved:
        #     raise Exception("How can this be??")
        print(q)

        precision_50_l.append(precision(qrels, retrieved, 50))
        precision_20_l.append(precision(qrels, retrieved, 20))
        precision_10_l.append(precision(qrels, retrieved, 10))
        precision_1_l.append(precision(qrels, retrieved, 1))
        recall_50_l.append(recall(qrels, retrieved, 50))
        recall_20_l.append(recall(qrels, retrieved, 20))
        recall_10_l.append(recall(qrels, retrieved, 10))
        recall_1_l.append(recall(qrels, retrieved, 1))

    with open(os.path.join(directory_path,'data','precision_50_l.bin'),'wb') as f:
        pickle.dump(precision_50_l, f)
    with open(os.path.join(directory_path,'data','precision_20_l.bin'),'wb') as f:
        pickle.dump(precision_20_l, f)
    with open(os.path.join(directory_path,'data','precision_10_l.bin'),'wb') as f:
        pickle.dump(precision_10_l, f)
    with open(os.path.join(directory_path,'data','precision_1_l.bin'),'wb') as f:
        pickle.dump(precision_1_l, f)

    with open(os.path.join(directory_path,'data','recall_50_l.bin'),'wb') as f:
        pickle.dump(recall_50_l, f)
    with open(os.path.join(directory_path,'data','recall_20_l.bin'),'wb') as f:
        pickle.dump(recall_20_l, f)
    with open(os.path.join(directory_path,'data','recall_10_l.bin'),'wb') as f:
        pickle.dump(recall_10_l, f)
    with open(os.path.join(directory_path,'data','recall_1_l.bin'),'wb') as f:
        pickle.dump(recall_1_l, f)

    et = time.time()
    print(f"Evaluation took {round(et-st,4)} seconds.") # 2959.9543 seconds
    

def eval_graph(num_queries):
    # with open(os.path.join(directory_path,'data','precision_50_l.bin'),'rb') as f:   
    #     precision_50_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','precision_20_l.bin'),'rb') as f:
        precision_20_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','precision_10_l.bin'),'rb') as f:
        precision_10_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','precision_1_l.bin'),'rb') as f:  
        precision_1_l = pickle.load(f)

    with open(os.path.join(directory_path,'data','recall_50_l.bin'),'rb') as f:
        recall_50_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','recall_20_l.bin'),'rb') as f:
        recall_20_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','recall_10_l.bin'),'rb') as f:
        recall_10_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','recall_1_l.bin'),'rb') as f:
        recall_1_l = pickle.load(f)
    # print(precision_1_l)
    # print(recall_1_l)

    X = np.arange(0,num_queries,1)
    # plot(X, precision_50_l, np.average(precision_50_l),  'precision@50', 1, Ylabel='precision')
    plot(X, precision_20_l, np.average(precision_20_l),  'precision@20', 2, Ylabel='precision')
    plot(X, precision_10_l, np.average(precision_10_l),  'precision@10', 3, Ylabel='precision')
    plot(X, precision_1_l,  np.average(precision_1_l),   'precision@1', 4, Ylabel='precision')
    plot(X, recall_50_l,    np.average(recall_50_l),     'recall@50', 5, Ylabel='recall')
    plot(X, recall_20_l,    np.average(recall_20_l),     'recall@20', 6, Ylabel='recall')
    plot(X, recall_10_l,    np.average(recall_10_l),     'recall@10', 7, Ylabel='recall')
    plot(X, recall_1_l,     np.average(recall_1_l),      'recall@1',  8, Ylabel='recall')
# hits([34,53,1,46,5,6,2,73,23,41],[651,1,6,66,4,58,5,64,168,41,38,486,51])
# corpus_embeddings_path = os.path.join(directory_path,'data','corpus_bert_embeddings.bin')
# queries_embeddings_path = os.path.join(directory_path,'data','queries_bert_embeddings.bin')
# relevance_path = os.path.join(directory_path,'data','queries_songs_relevance.bin')
# evaluate_full_dataset(corpus_embeddings_path, queries_embeddings_path,relevance_path, 1, 10)
# print("efghjky6u7ilouyjtrew")
eval_graph(3802)
""" AUC-ROC, and mean average precision (mAP)"""
"""Throughout the section, we use the standard
retrieval metrics: recall at rank k (R@k) which measures the
percentage of targets retrieved within the top k ranked results
(higher is better), along with the median (medR) and mean
(meanR) rank. For all metrics, we report the mean and standard
deviation of three different randomly seeded runs"""

def get_relevance_label_df(query_answer_pair_filepath):
    query_answer_pair = load_from_json(query_answer_pair_filepath)
    relevance_label_df = pd.DataFrame.from_records(query_answer_pair)
    return relevance_label_df

def get_relevance_label(relevance_label_df):
    relevance_label_df.rename(columns={'question': 'query_string'}, inplace=True)
    relevance_label = relevance_label_df.groupby(['query_string'])['answer'].apply(list).to_dict()
    return relevance_label

class Evaluation:
    """ Class for generating evaluation of re-ranked results 
    
    :param qas_filename: evaluating test queries using query_answer_pairs.json / synthetic_query_answer_pairs.json file
    
    :param rank_results_filepath: filepath to rank results
        BERT-FAQ/data/CovidFAQ/rank_results         # CovidFAQ
        BERT-FAQ/data/StackFAQ/rank_results         # StackFAQ
        BERT-FAQ/data/FAQIR/rank_results            # FAQIR
    :param jc_threshold: jaccard similarity threshold
    :param test_data: param used for generating evaluation for synthetic/user_query test data
    :param rankers: rankers e.g. unsupervised, supervised
    :param rank_fields: rank fields e.g. BERT-Q-a, BERT-Q-q
    :param loss_types:  loss types e.g. triplet, softmax
    :param query_types: query types e.g. faq, user_query
    :param neg_types: negative types e.g. simple, hard
    :param top_k: top k e.g. 2, 3, 5
    """

    def __init__(self, qas_filename, rank_results_filepath, jc_threshold=1.0, test_data="synthetic", rankers=["unsupervised", "supervised"], 
                 rank_fields=["BERT-Q-a", "BERT-Q-q"], loss_types=["triplet", "softmax"], query_types=["faq", "user_query"], 
                 neg_types=["simple", "hard"], top_k=[2, 3, 5, 10]):
        

        self.top_k = top_k
        # self.rankers = rankers
        self.neg_types = neg_types
        # self.test_data = test_data
        self.loss_types = loss_types
        # self.rank_fields = rank_fields
        # self.query_types = query_types
        self.rank_results_filepath = rank_results_filepath

        self.ndcg_per_query = []
        self.prec_per_query = []
        self.map_per_query = []

        list_of_qas = load_from_json(qas_filename)

        total_questions = 0
        filtered_questions = 0

        self.valid_queries = []
        
        for item in list_of_qas:
            total_questions += 1
            if 'jc_sim' in item:
                jc = float(item['jc_sim'])
                if jc <= jc_threshold:
                    filtered_questions += 1
                    self.valid_queries.append(item['question'])

    def compute_map(self, result_filepath, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute average precision score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_ap = 0
        num_queries = 0
        map_per_query = []
        for result in query_results:
            query_string = result['query_string']

            if query_string in self.valid_queries:

                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.array(labels)
                scores = np.array(reranks)

                ap = 0
                all_zeros = not np.any(labels)
                if labels and reranks and not all_zeros:
                    ap = average_precision_score(true_relevance, scores)

                sum_ap = sum_ap + ap
                num_queries = num_queries + 1

                query_map = {
                    "Query": query_string,
                    "MAP": ap,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                map_per_query.append(query_map)

        return (float(sum_ap / num_queries)), map_per_query

    def compute_prec(self, result_filepath, k, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute precision score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_prec = 0
        num_queries = 0
        prec_per_query = []
        for result in query_results:
            query_string = result['query_string']

            if query_string in self.valid_queries:
                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results[:k]:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.array(labels)
                scores = np.array(reranks)

                prec = 0
                all_zeros = not np.any(labels)
                if labels and reranks and not all_zeros:
                    prec = sum(true_relevance) / len(true_relevance)

                sum_prec = sum_prec + prec
                num_queries = num_queries + 1

                query_prec = {
                    "Query": query_string,
                    "k": k,
                    "Prec": prec,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                prec_per_query.append(query_prec)

        return (float(sum_prec / num_queries)), prec_per_query

    def compute_ndcg(self, result_filepath, k, ranker, match_field, rank_field="", loss_type="", query_type="", neg_type=""):
        """ Compute NDCG score for a set of rank results
        
        :param result_filepath:  filepath to Elasticsearch rank results
        :param ranker: supervised / unsupervised
        :param match_field: answer / question / question_answer / question_answer_concat
        :param loss_type: triplet / softmax
        :param rank_field: BERT-Q-a / BERT-Q-q
        :param query_type: faq / user_query
        :param neg_type: simple / hard
        """
        query_results = load_from_json(result_filepath)

        sum_ndcg = 0
        num_queries = 0
        ndcg_per_query = []

        for result in query_results:
            query_string = result['query_string']
         
            if query_string in self.valid_queries:
                
                topk_results = result['rerank_preds']

                labels = []
                reranks = []

                for topk in topk_results[:k]:
                    labels.append(topk['label'])
                    reranks.append(topk['score'])

                true_relevance = np.asarray([labels])
                scores = np.asarray([reranks])
                
                ndcg = 0
                if labels and reranks:
                    ndcg = ndcg_score(true_relevance, scores)

                sum_ndcg = sum_ndcg + ndcg
                num_queries = num_queries + 1

                query_ndcg = {
                    "Query": query_string,
                    "k": k,
                    "NDCG": ndcg,
                    "Method": ranker,
                    "Matching Field": match_field,
                    "Ranking Field": rank_field,
                    "Loss": loss_type,
                    "Training Data": query_type,
                    "Negative Sampling": neg_type
                }
                ndcg_per_query.append(query_ndcg)

        return (float(sum_ndcg / num_queries)), ndcg_per_query
    
