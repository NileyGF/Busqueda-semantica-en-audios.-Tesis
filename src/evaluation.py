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

def hit_rate(qrels: list, retrieved: list, top_k:int = 0):
    """ Number of retrieved relevant songs. Returns how many elements of qrels are in retrieved, using numpy 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        top_k:       if top_k > 0, only the top_k retrieved songs are considered
    """
    k = top_k if top_k > 0 else len(retrieved)
    considered = retrieved[:k]
    intersect = np.intersect1d(qrels, considered)
    # print(intersect)
    return 1 if len(intersect) > 0 else 0

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

def recall(qrels: list, retrieved: list, top_k:int = 0):
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

def average_precision(qrels: list, retrieved: list, top_k:int = 0):
    """ Average Precision is the average of the Precision scores computed after each relevant song is retrieved. 
        qrels:       list of relevant songs
        retrieved:   list of retrieved songs
        top_k:       if top_k > 0 only the top_k retrieved songs are considered
        
        Average Precision = sum^n_k (P@k * r_k) / R
        r_k:    is 1 if retrieved[k] is relevant, 0 otherwise
        R:      number of relevant songs
        n:      number of retrieved songs
    """
    # https://vitalflux.com/mean-average-precision-map-for-information-retrieval-systems/
    ap = []
    topk = top_k if top_k > 0 else len(retrieved)
    for k in range(topk):
        if retrieved[k] in qrels:
            pk = precision(qrels, retrieved, k+1)
            # print("pk: ", pk, "rel: ", rel)
            ap.append(pk)
            # print(ap)

    if len(ap) == 0:
        return 0
    return np.sum(ap) / len(ap)
# print(average_precision([0,1,3,4,6,9],[0,1,2,3,4,5,6,7,8,9]))

def mean_average_precision(average_precision:list):
    
    return np.mean(average_precision)
    return np.sum(average_precision) / len(average_precision)


def _evaluate(corpus_embedd:str, queries_embedd:str, relevance:str, metrics:str, top_k:int ):
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
        
    np.random.seed(0) 
    queries_indexes = list(np.random.permutation(np.arange(0,len(queries_embeddings_list))) [:1000])
    random_1k_queries = {i:queries_embeddings_list[i] for i in queries_indexes}
    st = time.time()
    metrics_d = {m:[] for m in metrics}
    for q in random_1k_queries:
        retrieved = emb_ret.evaluate_query(query_vector=random_1k_queries[q], documents_vectors_list=corpus_embeddings_list, top_k='all')
        qrels = relevance_list[q]
        # if q not in retrieved:
        #     raise Exception("How can this be??")
        # print(q)
        if 'precision' in metrics:
            metrics_d['precision'].append(precision(qrels, retrieved, top_k))
        if 'R@1' in metrics:
            metrics_d['R@1'].append(recall(qrels, retrieved, 1))
        if 'R@5' in metrics:
            metrics_d['R@5'].append(recall(qrels, retrieved, 5))
        if 'R@10' in metrics:
            metrics_d['R@10'].append(recall(qrels, retrieved, 10))
        if 'R@50' in metrics:
            metrics_d['R@50'].append(recall(qrels, retrieved, 50))
        if 'recall' in metrics:
            metrics_d['recall'].append(recall(qrels, retrieved, top_k))
        if 'average_precision' in metrics:
            metrics_d['average_precision'].append(average_precision(qrels, retrieved))
        if 'average_precision@10' in metrics:
            metrics_d['average_precision@10'].append(average_precision(qrels, retrieved, 10))
    et = time.time()
    print(f"Evaluation took {round(et-st,4)} seconds.") # over 1021 seconds /128 sec
    return metrics_d

def full_evaluate(restart=True):
    evals_df_path = os.path.join(directory_path,'data','evaluations.csv')
    idxs = ["captions_descriptions", "captions_descriptions_extend", "captions_tags_descriptions", "tags_descriptions", 
            "tags_descriptions_extend", "tags_tags_descriptions", "MuLan", "MusCALL", "Contrastive_SBERT", "Triplet_SBERT"]
    embedd_directory = os.path.join(directory_path,'data','embeddings')
    embeddings_files = {"captions_descriptions":(os.path.join(embedd_directory,'queries_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries_songs_relevance.bin')), 
                        "captions_tags_descriptions":(os.path.join(embedd_directory,'queries_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus2_bert_embeddings.bin'),os.path.join(directory_path,'data','queries_songs_relevance.bin')), 
                        "captions_descriptions_extend":(os.path.join(embedd_directory,'queries_bert_embeddings.bin'), os.path.join(embedd_directory,'extended_corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries_songs_relevance.bin')), 
                        "tags_descriptions":(os.path.join(embedd_directory,'queries2_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries2_songs_relevance.bin')), 
                        "tags_tags_descriptions":(os.path.join(embedd_directory,'queries2_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus2_bert_embeddings.bin'),os.path.join(directory_path,'data','queries2_songs_relevance.bin')), 
                        "tags_descriptions_extend":(os.path.join(embedd_directory,'queries2_bert_embeddings.bin'), os.path.join(embedd_directory,'extended_corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries2_songs_relevance.bin'))
                        }
    cols = ['R@1', 'R@5', 'R@10', 'R@50','mAP@10','mAP']
    if restart:
        # options_metrics_dict = {"captions_descriptions":[],
        #                         "captions_descriptions_extend":[],
        #                         "captions_tags_descriptions":[],
        #                         "tags_descriptions":[],
        #                         "tags_descriptions_extend":[],
        #                         "tags_tags_descriptions":[],
        #                         "MuLan":                      [None, None, None, None,None,0.084],
        #                         "MusCALL":                    [0.259,0.519,0.633,None,0.36,None],
        #                         "Contrastive_SBERT":          [0.068,0.254,0.384,None,0.15,None],
        #                         "Triplet_SBERT":              [0.067,0.236,0.366,None,0.14,None],
        #                         }
        # cols = ['R@1', 'R@5', 'R@10', 'R@50','mAP@10','mAP']
        options_metrics_dict = {'rows':idxs,
                                'R@1':[None,None,None,None,None,None,None,0.259,0.068,0.067], 
                                'R@5':[None,None,None,None,None,None,None,0.519,0.254,0.236], 
                                'R@10':[None,None,None,None,None,None,None,0.633,0.384,0.366], 
                                'R@50':[None,None,None,None,None,None,None,None,None,None], 
                                'mAP@10':[None,None,None,None,None,None,None,0.36,0.15,0.14],
                                'mAP':[None,None,None,None,None,None,0.084,None,None,None]}
        evals_df = pd.DataFrame.from_dict(options_metrics_dict)
        evals_df.to_csv(evals_df_path, index=False)
    else:
        evals_df = pd.read_csv(evals_df_path)
        options_metrics_dict = evals_df.to_dict(orient='list')
    
    for idx, row in evals_df.iterrows():
        """
        row['rows'] = query/corpus pair
        row['R@1'], row['R@5'], row['R@10'], row['R@50'], row[mAP@10] , row[mAP] 
        """
        if idx >= 6: break
        metrics = []
        if np.isnan(row['R@1']):
            metrics.append('R@1')
        if np.isnan(row['R@5']):
            metrics.append('R@5')
        if np.isnan(row['R@10']):
            metrics.append('R@10')
        if np.isnan(row['R@50']):
            metrics.append('R@50')
        if np.isnan(row['mAP@10']):
            metrics.append('average_precision@10')
        if np.isnan(row['mAP']):
            metrics.append('average_precision')
        print(metrics)
        
        if len(metrics) == 0: continue

        metrics_result = _evaluate(corpus_embedd=embeddings_files[row['rows']][1], 
                                    queries_embedd=embeddings_files[row['rows']][0],
                                    relevance=embeddings_files[row['rows']][2],
                                    metrics=metrics, top_k='all')
        for m in metrics_result:
            if np.mean(metrics_result[m]) > 0.00001:
                metrics_result[m] = round(np.mean(metrics_result[m]),5)
            else:
                metrics_result[m] = np.mean(metrics_result[m])
            print(m, metrics_result[m])
            if 'average_precision' == m:
                evals_df.at[idx,'mAP'] = metrics_result[m]
            elif 'average_precision@10' == m:
                evals_df.at[idx,'mAP@10'] = metrics_result[m]
            else: 
                evals_df.at[idx, m] = metrics_result[m]
        
        evals_df.to_csv(evals_df_path, index=False)
    evals_df.to_csv(evals_df_path, index=False)
    return evals_df

