import src.core as core
import src.features_extractors as features_extractors
import pandas as pd
import os

def process_song(file_path):
    song = core.Song(file_path)
    song_info = {}
    for feat in features_extractors.SPECIFIC_FEATURES:
        extractor:core.FeaturesExtractor = features_extractors.SPECIFIC_FEATURES[feat]()
        song_info[feat] = extractor.extract_feature(song)
    return song_info

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
music_folder = os.path.join(directory_path, 'music_examples')

features_dict = {'ytid':[]}
for feat in features_extractors.SPECIFIC_FEATURES:
    features_dict[feat] = []

for s in os.listdir(music_folder):
    # song = os.path.join(music_folder,os.listdir(music_folder)[0])
    # print(song)
    song = os.path.join(music_folder,s)
    features_dict['ytid'].append(s)
    song_feat = process_song(song)
    for f in song_feat:
        features_dict[f].append(song_feat[f])
    
    features_df = pd.DataFrame.from_dict(features_dict)
    features_df.to_csv(os.path.join(directory_path,'data','test-feat_index.csv'), index=False) # , header=False

def process_all(music_folder):

    for file in os.listdir(music_folder):
        song = os.path.join(music_folder, file)
        print(song)
        print(process_song(song))
        