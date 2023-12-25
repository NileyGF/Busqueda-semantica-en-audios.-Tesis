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


def plot(X:list, Y:list, avg: float, name:str, fig: int, Xlabel = 'queries', Ylabel=''):
    # path = os.path.join(os.path.dirname(__file__), 'evaluation')
    # path = os.path.join(path, name)
    plt.figure(fig)
    plt.scatter(X, Y)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title('Average '+name+': '+str(round(avg,4)))
    plt.ylim([-0.1, 1.1])
    # plt.savefig(path, format='png')
    plt.show()

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
    print(f"Evaluation took {round(et-st,4)} seconds.") # over 3600 seconds
    return metrics_d

def __evaluate(corpus_embedd:str, queries_embedd:str, relevance:str):
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

def full_evaluate(restart=True):
    evals_df_path = os.path.join(directory_path,'data','evaluations.csv')
    idxs = ["captions_descriptions", "captions_descriptions_extend", "tags_descriptions", 
            "tags_descriptions_extend", "MuLan", "SoundDescs",
            "LangBasedRetrieval_SentenceBERT","Contrastive_SentenceBERT"]
    embedd_directory = os.path.join(directory_path,'data','embeddings')
    embeddings_files = {"captions_descriptions":(os.path.join(embedd_directory,'queries_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries_songs_relevance.bin')), 
                        "captions_descriptions_extend":(os.path.join(embedd_directory,'queries_bert_embeddings.bin'), os.path.join(embedd_directory,'extended_corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries_songs_relevance.bin')), 
                        "tags_descriptions":(os.path.join(embedd_directory,'queries2_bert_embeddings.bin'), os.path.join(embedd_directory,'corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries2_songs_relevance.bin')), 
                        "tags_descriptions_extend":(os.path.join(embedd_directory,'queries2_bert_embeddings.bin'), os.path.join(embedd_directory,'extended_corpus_bert_embeddings.bin'),os.path.join(directory_path,'data','queries2_songs_relevance.bin'))
                        }
    cols = ['R@1', 'R@5', 'R@10', 'R@50','mAP@10','mAP']
    if restart:
        # options_metrics_dict = {"captions_descriptions":[],
        #                         "captions_descriptions_extend":[],
        #                         "tags_descriptions":[],
        #                         "tags_descriptions_extend":[],
        #                         "MuLan":[None,None,None,None,0.084],
        #                         "SoundDescs":[31.1,60.6,70.8,86,None],
        #                         "MusCALL":[25.9,51.9,63.4,None,None],
        #                         "LangBasedRetrieval_SentenceBERT":[0.04,0.16,0.25,None,None],
        #                         "Contrastive_SentenceBERT":[6.8,25.4,38.4,None,None]
        #                         }
        # cols = ['R@1', 'R@5', 'R@10', 'R@50','mAP@10','mAP']
        options_metrics_dict = {'rows':idxs,
                                'R@1':[None,None,None,None,None,31.1,0.04,6.8], 
                                'R@5':[None,None,None,None,None,60.6,0.16,25.4], 
                                'R@10':[None,None,None,None,None,70.8,0.25,38.4], 
                                'R@50':[None,None,None,None,None,86,None,None], 
                                'mAP@10':[None,None,None,None,None,None,None,0],
                                'mAP':[None,None,None,None,0.081,None,None,None]}
        evals_df = pd.DataFrame.from_dict(options_metrics_dict)
        # evals_df.index = idxs
        evals_df.to_csv(evals_df_path, index=False)
        # print(evals_df)
    else:
        evals_df = pd.read_csv(evals_df_path)
        # print(evals_df)
        options_metrics_dict = evals_df.to_dict(orient='list') # ,index=False
        # print(options_metrics_dict)
    
    for idx, row in evals_df.iterrows():
        """
        row['rows'] = query/corpus pair
        row['R@1'], row['R@5'], row['R@10'], row['R@50'], row[mAP] 
        """
        if idx >= 4: break
        # print(idx,row['R@1'])
        # print(evals_df.at[idx, 'R@1'])
        metrics = []
        # print(row)
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
        # metrics_result = {'R@10':[1244,6,57,6,8,3,3,72,72],
        #                     'R@50':[534,354,6,24,1,3,5,4,36,5,7],
        #                     'average_precision':[1,2,3,4,5,6]}
        for m in metrics_result:
            metrics_result[m] = round(np.mean(metrics_result[m]),3)
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

def eval_graph(num_queries):
    # with open(os.path.join(directory_path,'data','regular_eval','precision_50_l.bin'),'rb') as f:   
    #     precision_50_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','precision_20_l.bin'),'rb') as f:
        precision_20_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','precision_10_l.bin'),'rb') as f:
        precision_10_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','precision_1_l.bin'),'rb') as f:  
        precision_1_l = pickle.load(f)

    with open(os.path.join(directory_path,'data','regular_eval','recall_50_l.bin'),'rb') as f:
        recall_50_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','recall_20_l.bin'),'rb') as f:
        recall_20_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','recall_10_l.bin'),'rb') as f:
        recall_10_l = pickle.load(f)
    with open(os.path.join(directory_path,'data','regular_eval','recall_1_l.bin'),'rb') as f:
        recall_1_l = pickle.load(f)
    # print(precision_1_l)
    # print(recall_1_l)

    X = np.arange(0,num_queries,1)
    # plot(X, precision_50_l, np.average(precision_50_l),  'precision@50', 1, Ylabel='precision')
    # plot(X, precision_20_l, np.average(precision_20_l),  'precision@20', 2, Ylabel='precision')
    # plot(X, precision_10_l, np.average(precision_10_l),  'precision@10', 3, Ylabel='precision')
    # plot(X, precision_1_l,  np.average(precision_1_l),   'precision@1', 4, Ylabel='precision')
    # plot(X, recall_50_l,    np.average(recall_50_l),     'recall@50', 5, Ylabel='recall')
    # plot(X, recall_20_l,    np.average(recall_20_l),     'recall@20', 6, Ylabel='recall')
    # plot(X, recall_10_l,    np.average(recall_10_l),     'recall@10', 7, Ylabel='recall')
    # plot(X, recall_1_l,     np.average(recall_1_l),      'recall@1',  8, Ylabel='recall')


