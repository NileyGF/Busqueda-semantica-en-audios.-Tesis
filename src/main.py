from features_extractors import SPECIFIC_FEATURES
from core import Song, FeaturesExtractor
import pandas as pd
import os

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
musiccaps_csv_all_path =  os.path.join(directory_path, "musiccaps-public.csv")

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
    return features

def extract_features_all_dataset():
    """ Go through the dataset to extract the features.  """

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