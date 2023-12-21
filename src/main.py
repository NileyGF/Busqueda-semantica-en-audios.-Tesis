from src.features_extractors import SPECIFIC_FEATURES
import src.embedding_retrieval as emb_ret
from src.core import Song, FeaturesExtractor, musiccaps_preprocess, directory_path, downloaded_songs_name_path
import os
from pathlib import Path
from ast import literal_eval
import pandas as pd
import pickle
import time
import nltk


""" First part of the pipeline : 
Feature extraction of a music
Extensible model that goes through all the features extractors and creates a dictionary {feature: value} 
Then, save it, either with pickle or do a pandas table.
"""
def extract_features(music_path:str, features_dict:dict=SPECIFIC_FEATURES) -> dict :
    """ Given a music file path, extract all the features in features_dict and return a dictionary {feature: value}.
        features_dict : dict -> feature_name: feature_extractor_class
    """
    song = Song(music_path)
    features = {}
    for feat in features_dict:
        try:
            extractor:FeaturesExtractor = features_dict[feat]()
            features[feat] = extractor.extract_feature(music=song)
        except Exception as err:
            print("Error while extracting features: ", err)
            features[feat] = None
    return features

def extract_features_all_dataset() -> pd.DataFrame:
    """ Go through the dataset to extract the features.  """
    try: 
        dataset_path = os.path.join(directory_path,'data','musiccaps-subset.csv') # 'musiccaps-subset.csv'
        musiccaps_df = pd.read_csv(dataset_path)
    except:
        musiccaps_preprocess()
        dataset_path = os.path.join(directory_path,'data','musiccaps-subset.csv') # 'musiccaps-subset.csv'
        musiccaps_df = pd.read_csv(dataset_path)
    songs_dict = downloaded_songs_name_path()

    try: 
        file = open(os.path.join(directory_path,'data','temp_feat_dict.bin'),'rb')
        features_dict = pickle.load(file)
        file.close()
    except Exception as er:
        print(er)
        features_dict = {'ytid':[]}
        for feat in SPECIFIC_FEATURES:
            features_dict[feat] = []

    processed = len(features_dict['ytid'])
    print(processed) # 3805 18 horas
    # return
    for song in musiccaps_df.itertuples():
        # print(song, song.Index)
        if song.Index < processed:
            continue
        print("Extracting features from ",songs_dict[song.ytid])
        features_dict['ytid'].append(song.ytid)
        song_feat = extract_features(songs_dict[song.ytid])
        for f in song_feat:
            if song_feat[f] == None:
                features_dict[f].append("")
            else:
                features_dict[f].append(song_feat[f])
        with open(os.path.join(directory_path,'data','temp_feat_dict.bin'),'wb') as file:
            pickle.dump(features_dict,file)
    
    features_df = pd.DataFrame.from_dict(features_dict)

    inner_merged = pd.merge(musiccaps_df, features_df)
    # print(inner_merged.head())
    # print()
    # print(inner_merged.shape, musiccaps_df.shape, features_df.shape)

    # inner_merged.to_csv(os.path.join(directory_path,'data','musiccaps-subset-feat_index.csv')) # , header=False, index=False
    inner_merged.to_csv(os.path.join(directory_path,'data','musiccaps-subset-feat.csv'), index=False) 
    return inner_merged    
        
def extract_new_features_all_dataset(new_features_dict:dict, earlier_df:str="musiccaps-subset-feat") -> pd.DataFrame:
    """ Go through the dataset to extract the features in new_features_dict.  
        new_features_dict : dict -> feature_name: feature_extractor_class.
        earlier_df : str -> the name of the dataframe with the previous features extracted.
    """
    songs_dict = downloaded_songs_name_path()    
    musiccaps_df = pd.read_csv(os.path.join(directory_path,'data',earlier_df+'.csv'))
    try: 
        file = open(os.path.join(directory_path,'data','temp_feat_dict.bin'),'rb')
        features_dict = pickle.load(file)
        file.close()
    except Exception as er:
        print(er)
        features_dict = {'ytid':[]}
        for feat in new_features_dict:
            features_dict[feat] = []

    processed = len(features_dict['ytid'])

    for song in musiccaps_df.itertuples():
        if song.index < processed:
            continue
        if song.index == 10:
            return features_dict
        print("Extracting features from ",songs_dict[song.ytid])
        features_dict['ytid'].append(song.ytid)
        song_feat = extract_features(music_path=songs_dict[song.ytid], features_dict=new_features_dict)
        for f in song_feat:
            if song_feat[f] == None:
                features_dict[f].append("")
            else:
                features_dict[f].append(song_feat[f])
        with open(os.path.join(directory_path,'data','temp_feat_dict.bin'),'wb') as file:
            pickle.dump(features_dict,file)
    
    features_df = pd.DataFrame.from_dict(features_dict)

    inner_merged = pd.merge(musiccaps_df, features_df)
    # print(inner_merged.head())
    # print()
    # print(inner_merged.shape, musiccaps_df.shape, features_df.shape)

    # inner_merged.to_csv(os.path.join(directory_path,'data',earlier_df+'_index.csv')) # , header=False, index=False
    inner_merged.to_csv(os.path.join(directory_path,'data',earlier_df+'.csv'), index=False) 
    return inner_merged   

def update_a_song_features(index:int, features_dict:dict=SPECIFIC_FEATURES):
    
    dataset_path = os.path.join(directory_path,'data','musiccaps-subset-feat.csv') 
    musiccaps_df = pd.read_csv(dataset_path)
    songs_dict = downloaded_songs_name_path()

    song_path = songs_dict[musiccaps_df.at[index,'ytid']]
    print(song_path)
    
    song_feat = extract_features(music_path=song_path, features_dict=features_dict)
    for f in song_feat:
        musiccaps_df.at[index,f] = song_feat[f]

    musiccaps_df.to_csv(dataset_path, index=False)
    return musiccaps_df

# error_songs=["-cQ-jUTEgck","3OLeJZF4oI0","5xBfKiQcMZQ","6k6tF0s0lNs","6lPgzqrvHHw","6lPw0wKu7_M","6nWWTNVRDjw","FXVu-YwjhxM",
#              "IL1n6jzABVw","ILE12hEW5Ck","IN71kMOAk_k","VRxFYwbik6A","VV85n-ebuUU","VZJfkEet6EQ","VZfrDZhI7BU","crToMuDxmL8",
#              "ewI4q7_4TBQ","hRbukCd6N68","hSSzn4bIwZg","hTAWbHXCJ2A","hTNKYJ6suII","hTPxqUtlLdo","pLqasxGYWX8","rLtXML8Y5wo" ]
# dataset_path = os.path.join(directory_path,'data','musiccaps-subset.csv') 
# musiccaps_df = pd.read_csv(dataset_path)
# musiccaps_df= musiccaps_df[~musiccaps_df['ytid'].isin(error_songs)]
# print(musiccaps_df.columns)
# musiccaps_df = musiccaps_df.drop(['Unnamed: 0'], axis=1)
# print(musiccaps_df.columns)
# musiccaps_df.to_csv(os.path.join(directory_path,'data','musiccaps-subset.csv'), index=False) 
# extract_features_all_dataset()
# processed_dataset_path = os.path.join(directory_path,'musiccaps-subset-feat.csv') # os.path.join(directory_path,'musiccaps-subset-feat_index.csv')


""" Second part of the pipeline : 
Convert the features information from tags (metadata), to a sentence, caption like.
Now I'm doing static descriptor per-feature
# Temporarily it will be approached using GPT2 model (huggingface API) for complete sentences.
# It is necessary to devise a prompt that maximizes the information fidelity. 
# It is possible to do some fine-tunnig if it is decided to download the GPT model instead of only using an API. 
"""
def get_descriptions_from_feat(features_dict:list, df_name:str='musiccaps-subset-feat.csv'):
    """ Given a features dictionary and a dataframe and get a descriptions for every song in the dataframe, 
        using the features that are in the dictionary. Also, the description for each feaure will be extracted 
        using the property 'feature_description' from the FeaturesExtractor.
        
        features_dict : dict -> feature_name: feature_extractor_class
        df_name : str -> .csv with the songs and the features extracted
    """
    st = time.time()
    dataset_path = os.path.join(directory_path,'data',df_name) 
    musiccaps_df = pd.read_csv(dataset_path)
    
    description_dict = {'ytid':[], 'description':[]}
    # descriptions_and_name = []
    # descriptions = []

    for idx, row in musiccaps_df.iterrows():
        song_description = ""
        description_dict['ytid'].append(row['ytid'])
        # print(idx)
        for feat in features_dict:
            extractor: FeaturesExtractor = features_dict[feat]
            feature_value = row[feat]
            song_description += extractor.feature_description(feature=feature_value)
        description_dict['description'].append(song_description)
        
    features_df = pd.DataFrame.from_dict(description_dict)

    inner_merged = pd.merge(musiccaps_df, features_df)
    # inner_merged.to_csv(os.path.join(directory_path,'data','musiccaps-subset-desriptions_index.csv')) # , header=False, index=False
    inner_merged.to_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'), index=False) 
    
    with open(os.path.join(directory_path,'data','descriptions_list.bin'),'wb') as file:
        pickle.dump(description_dict['description'], file)
    print(round(time.time()-st,4)," sec")
    # time: 0.75 sec
    # return descriptions, descriptions_and_name
    return description_dict['description'], inner_merged   

# get_descriptions_from_feat(SPECIFIC_FEATURES)

""" Third part of the pipeline : 
Information Retrieval System (Temporarily is ccosine similarity )
Using BERT, extract embeddings for each song sentence and use that as the vectors used in the comparison with the query.
The BERT model will need to be downloaded.
Decide on an appropiate rank K as the max number of relevant results. (Only retrieve the top k most similars)
"""
def get_queries2(tags_list:list) -> list:
    queries = []
    for s in tags_list:
        song_tags = literal_eval(s)
        query = ""
        for t in song_tags:
            query += t + ', '
        queries.append(query)
    
    return queries

def save_queries2_embedd():
    try: 
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    except Exception as er:
        print(er)
        get_descriptions_from_feat(SPECIFIC_FEATURES)
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    queries_list = descriptions_df["aspect_list"].values.tolist()
    embeddings_path = os.path.join(directory_path,'data','embeddings','queries2_bert_embeddings.bin')

    st = time.time()
    print('starting:',st)
    
    queries_list = get_queries2(queries_list)
    emb_ret.extract_embeddings_for_docs_list(documents_list=queries_list, save=True, save_path=embeddings_path)
    with open(embeddings_path, 'rb') as f:
        queries_embeddings_list = pickle.load(f)
    et = time.time()
    print(f"Embeddings extractions for the queries 2 using BERT, took {round(et-st,4)} seconds.") # 5292.654 seconds 
    return embeddings_path, queries_embeddings_list

def save_descript_embedd():
    try: 
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    except Exception as er:
        print(er)
        get_descriptions_from_feat(SPECIFIC_FEATURES)
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    
    descriptions_list = descriptions_df["description"].values.tolist()
    embeddings_path = os.path.join(directory_path,'data','embeddings','corpus_bert_embeddings.bin')
    
    st = time.time()
    print('starting:',st)

    emb_ret.extract_embeddings_for_docs_list(documents_list=descriptions_list, save=True, save_path=embeddings_path)
    with open(embeddings_path, 'rb') as f:
        docs_embeddings_list = pickle.load(f)

    embedd_corpus_relation_path = os.path.join(directory_path,'data','corpus-embeddings_rel.bin')
    embedd_corpus_relation = [(i,i) for i in range(len(docs_embeddings_list))]
    with open(embedd_corpus_relation_path, 'wb') as f:
        pickle.dump(embedd_corpus_relation, f)

    et = time.time()
    print(f"Embeddings extractions for the corpus using BERT, took {round(et-st,4)} seconds.") # 5097.6766 seconds
    return embeddings_path, docs_embeddings_list, embedd_corpus_relation

def extended_descript_embedd():
    descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    descriptions_list = descriptions_df["description"].values.tolist()
    embeddings_path = os.path.join(directory_path,'data','embeddings','corpus_bert_embeddings.bin')
    embedd_corpus_relation_path = os.path.join(directory_path,'data','corpus-embeddings_rel.bin')
    with open(embeddings_path, 'rb') as f:
        embeddings_list = pickle.load(f)
    with open(embedd_corpus_relation_path, 'rb') as f:
        embedd_corpus_relation = pickle.load(f)
    st = time.time()
    print('starting:',st)
    last = len(embedd_corpus_relation)
    for s in range(len(descriptions_list)):
        sentences = nltk.sent_tokenize(descriptions_list[s])
        print(s)
        for i, text in enumerate(sentences):
            tokenized = emb_ret.BERT_embedding.bert_tokenize(text=text)
            if len(tokenized) > 512:
                print(f"The BERT tokens, for the text number {i}, length is longer than the specified maximum sequence length . {len(tokenized)} > 512. ")
                print(text)
                raise Exception()
            embedding, _ = emb_ret.BERT_embedding.sentential_embeddings(tokenized_text=tokenized)
            embeddings_list.append(embedding)
            embedd_corpus_relation.append((last,s))
            last += 1
    extended_embeddings_path = os.path.join(directory_path,'data','embeddings','corpus_bert_embeddings.bin')
    with open(extended_embeddings_path, 'wb') as f:
        pickle.dump(embeddings_list,f)
    with open(embedd_corpus_relation_path, 'wb') as f:
        pickle.dump(embedd_corpus_relation,f) 
    et = time.time()
    print(f"Embeddings extractions for the corpus using BERT, took {round(et-st,4)} seconds.") # 52863.9011 seconds
    return extended_embeddings_path, embeddings_list, embedd_corpus_relation


def save_queries_embedd():
    try: 
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    except Exception as er:
        print(er)
        get_descriptions_from_feat(SPECIFIC_FEATURES)
        descriptions_df = pd.read_csv(os.path.join(directory_path,'data','musiccaps-subset-descriptions.csv'))
    queries_list = descriptions_df["caption"].values.tolist()
    embeddings_path = os.path.join(directory_path,'data','embeddings','queries_bert_embeddings.bin')

    st = time.time()
    print('starting:',st)

    emb_ret.extract_embeddings_for_docs_list(documents_list=queries_list, save=True, save_path=embeddings_path)
    with open(embeddings_path, 'rb') as f:
        queries_embeddings_list = pickle.load(f)
    et = time.time()
    print(f"Embeddings extractions for the queries using BERT, took {round(et-st,4)} seconds.") # 3855.854 seconds 
    return embeddings_path, queries_embeddings_list

def relevant_descriptions_by_query(query:str, top_k='all', embeddings_path='corpus_bert_embeddings.bin'):
    docs_embeddings_list = None
    documents_list = None
    try:
        with open(embeddings_path, 'rb') as f:
            docs_embeddings_list = pickle.load(f)
    except:
        documents_list = [] # TODO
        docs_embeddings_list = emb_ret.extract_embeddings_for_docs_list(documents_list=documents_list, save=True, save_path=embeddings_path)

    if docs_embeddings_list != None:
        docs_idx_list = emb_ret.process_query(query=query, docs_embeddings_list=docs_embeddings_list, top_k=top_k)
    else:
        docs_idx_list = emb_ret.process_query(query=query, documents_list=documents_list, top_k=top_k)

    return docs_idx_list

# save_queries2_embedd()
extended_descript_embedd()

""" Fourth part of the pipeline : 
Django web app to access and test the Information Retrieval System.
"""

""" Fifth part of the pipeline : 
Evaluations.
Using Recall-K, Acurracy-K, F1 and so; evaluate the information retrieval. Using ablation experiment 
with different SRI approaches, like whether embeddings improve over only vectorial. Use as queries the original 
captions from the dataset, and it should return at least the song, perhaps include similar songs as good results too.
Search for metrics to evaluate the sentence generation from table. 

Ablations:

"""
def most_similars_to_query(query:list, embeddings_list:list, min_sim=0.85):
    rel = []
    for k in range(len(embeddings_list)):
        cosine_sim = emb_ret.BERT_embedding.np_cosine_similarity(query, embeddings_list[k])
        # cosine_sim = BERT_embedding.cosine_distance(query_vector, documents_vectors_list[k])
        rel.append((k,cosine_sim))

    rel = sorted(rel, key=lambda item: item[1], reverse=True)
    i = 0
    while rel[i][1] >= min_sim:
        i += 1

    return rel[:i]

def relevance_judgments() -> list:
    """ for every query/caption find the other captions with similarity > 0.9; to set those songs as relevant.
    Returns a list R, where R[i] = List[Tuples[song_idx, similarity]]
    """
    queries_embeddings_path = os.path.join(directory_path,'data','embeddings','queries_bert_embeddings.bin')
    with open(queries_embeddings_path, 'rb') as f:
        queries_embeddings_list = pickle.load(f) 
    relevance = []
    
    st = time.time()
    print('starting:',st)
    for caption in queries_embeddings_list:
        similars = most_similars_to_query(caption,queries_embeddings_list,0.95)
        relevance.append(similars)
        # print(len(similars))

    relevance_path = os.path.join(directory_path,'data','queries_songs_relevance.bin')
    with open(relevance_path, 'wb') as f:
        pickle.dump(relevance,f)
    et = time.time()
    print(f"Relevance queries-songs calculation, took {round(et-st,4)} seconds.") # 370.1901 seconds 
    
    return relevance

# relevance_judgments()
