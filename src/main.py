from features_extractors import SPECIFIC_FEATURES
from core import Song, FeaturesExtractor, musiccaps_preprocess, directory_path, downloaded_songs_names
import os
import pandas as pd


""" First part of the pipeline : 
Feature extraction of a music
Extensible model that goes through all the features extractors and creates a dictionary {feature: value} 
Then, save it, either with pickle or do a pandas table.
"""
def extract_features(music_path:str) -> dict :
    """ Given a music file path, extract all the features in SPECIFIC_FEATURES and return a dictionary {feature: value} """
    song = Song(music_path)
    features = {}
    for feat in SPECIFIC_FEATURES:
        try:
            extractor:FeaturesExtractor = SPECIFIC_FEATURES[feat]()
            features[feat] = extractor.extract_feature(music=song)
        except Exception as err:
            print("Error while extracting features: ", err)
            features[feat] = None
    return features

def extract_features_all_dataset():
    """ Go through the dataset to extract the features.  """
    try: 
        dataset_path = os.path.join(directory_path,'musiccaps-subset_index.csv') # 'musiccaps-subset.csv'
        musiccaps_df = pd.read_csv(dataset_path)
    except:
        musiccaps_preprocess()
        dataset_path = os.path.join(directory_path,'musiccaps-subset_index.csv') # 'musiccaps-subset.csv'
        musiccaps_df = pd.read_csv(dataset_path)
    songs_dict = downloaded_songs_names()

    features_dict = {'ytid':[]}
    for feat in SPECIFIC_FEATURES:
        features_dict[feat] = []

    for song in musiccaps_df.itertuples():
        print(songs_dict[song.ytid])
        features_dict['ytid'].append(song.ytid)
        song_feat = extract_features(songs_dict[song.ytid])
        for f in song_feat:
            features_dict[f].append(song_feat[f])
    
    features_df = pd.DataFrame.from_dict(features_dict)

    inner_merged = pd.merge(musiccaps_df, features_df)
    # print(inner_merged.head())
    # print()
    # print(inner_merged.shape, musiccaps_df.shape, features_df.shape)

    inner_merged.to_csv(os.path.join(directory_path,'musiccaps-subset-feat_index.csv')) # , header=False, index=False
    inner_merged.to_csv(os.path.join(directory_path,'musiccaps-subset-feat.csv'), index=False) 
    return inner_merged    
        
extract_features_all_dataset()
processed_dataset_path = os.path.join(directory_path,'musiccaps-subset-feat.csv') # os.path.join(directory_path,'musiccaps-subset-feat_index.csv')


""" Second part of the pipeline : 
Convert the features information from tags (metadata), to a sentence, caption like.
Temporarily it will be approached using GPT2 model (huggingface API) for complete sentences.
It is necessary to devise a prompt that maximizes the information fidelity. 
It is possible to do some fine-tunnig if it is decided to download the GPT model instead of only using an API. 
"""

""" Third part of the pipeline : 
Information Retrieval System
Using BERT, extract embeddings for each song sentence and use that as the vectors used in the comparison with the query.
The BERT model will need to be downloaded.
Decide on an appropiate rank K as the max number of relevant results. (Only retrieve the top k most similars)
"""

""" Fourth part of the pipeline : 
Django web app to access and test the Information Retrieval System.
"""

""" Fifth part of the pipeline : 
Evaluations.
Using Recall-K, Acurracy-K, F1 and so; evaluate the information retrieval. Using ablation experiment 
with different SRI approaches, like whether embeddings improve over only vectorial. Use as queries the original 
captions from the dataset, and it should return at least the song, perhaps include similar songs as good results too.
Search for metrics to evaluate the sentence generation from table. 

"""