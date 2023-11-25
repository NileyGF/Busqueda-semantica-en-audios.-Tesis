import src.core as core
import src.features_extractors as features_extractors
import os

def process_song(file_path):
    song = core.Song(file_path)
    song_info = {}
    for feat in features_extractors.specific_features:
        extractor:core.FeaturesExtractor = features_extractors.specific_features[feat]()
        song_info[feat] = extractor.extract_feature(song)

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
music_folder = os.path.join(directory_path, 'music_examples')
song = os.path.join(music_folder,os.listdir(music_folder)[0])
# print(song)
print(process_song(song))

def process_all(music_folder):

    for file in os.listdir(music_folder):
        song = os.path.join(music_folder, file)
        print(song)
        print(process_song(song))
        