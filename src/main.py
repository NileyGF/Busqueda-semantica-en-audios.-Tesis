from src.features_extractors import SPECIFIC_FEATURES
# import src.embedding_retrieval as emb_ret
from src.core import Song, FeaturesExtractor, musiccaps_preprocess, directory_path, downloaded_songs_name_path
import os
from pathlib import Path
import pandas as pd
import pickle
import time


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
        dataset_path = os.path.join(directory_path,'data','musiccaps-subset_index.csv') # 'musiccaps-subset.csv'
        musiccaps_df = pd.read_csv(dataset_path)
    except:
        musiccaps_preprocess()
        dataset_path = os.path.join(directory_path,'data','musiccaps-subset_index.csv') # 'musiccaps-subset.csv'
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
    print(processed) # 3826 18 horas
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

    inner_merged.to_csv(os.path.join(directory_path,'data','musiccaps-subset-feat_index.csv')) # , header=False, index=False
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

    inner_merged.to_csv(os.path.join(directory_path,'data',earlier_df+'_index.csv')) # , header=False, index=False
    inner_merged.to_csv(os.path.join(directory_path,'data',earlier_df+'.csv'), index=False) 
    return inner_merged   

extract_features_all_dataset()
# processed_dataset_path = os.path.join(directory_path,'musiccaps-subset-feat.csv') # os.path.join(directory_path,'musiccaps-subset-feat_index.csv')


""" Second part of the pipeline : 
Convert the features information from tags (metadata), to a sentence, caption like.
Temporarily it will be approached using GPT2 model (huggingface API) for complete sentences.
It is necessary to devise a prompt that maximizes the information fidelity. 
It is possible to do some fine-tunnig if it is decided to download the GPT model instead of only using an API. 
"""
def temp_descriptions(df_name:str='musiccaps-subset-feat_index.csv'):
    dataset_path = os.path.join(directory_path,'data',df_name) 
    musiccaps_df = pd.read_csv(dataset_path)
    # print(df)
    # name_caption_df = df[["ytid", "caption"]]
    # print(name_caption_df)
    print(musiccaps_df.columns)
    for song in musiccaps_df.itertuples():
        print(song)
        break
    descriptions_and_name = []
    descriptions = []
    # for song in name_caption_df.itertuples():
    #     descriptions_and_name.append((song.ytid, song.caption)) 
    #     descriptions.append(song.caption)
    return descriptions, descriptions_and_name

# temp_descriptions()

""" Third part of the pipeline : 
Information Retrieval System (Temporarily is ccosine similarity )
Using BERT, extract embeddings for each song sentence and use that as the vectors used in the comparison with the query.
The BERT model will need to be downloaded.
Decide on an appropiate rank K as the max number of relevant results. (Only retrieve the top k most similars)
"""
def save_embedd():
    descriptions_list, descript_name_list = temp_descriptions() # TODO 
    embeddings_path = os.path.join(directory_path,'data','temp_corpus_bert_embeddings.bin')
    st = time.time()
    print('starting:',st)
    emb_ret.extract_embeddings_for_docs_list(documents_list=descriptions_list, save=True, save_path=embeddings_path)
    with open(embeddings_path, 'rb') as f:
        docs_embeddings_list = pickle.load(f)
    et = time.time()
    print(f"Embeddings extractions for the corpus using BERT, took {round(et-st,4)} seconds.")
    return embeddings_path, docs_embeddings_list

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