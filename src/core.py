# Huggingface token : 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
import pandas as pd
from pathlib import Path
import os

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path) #  src
musiccaps_csv_all_path =  os.path.join(directory_path, "musiccaps-public.csv")

class Song:
    def __init__(self, file_path:str) -> None:
        self.song_file = file_path
    
    def get_path(self) -> str:
        return self.song_file

class FeaturesExtractor:
    """ Common features extractor of music to stablish a single protocol for all the models that can join the system """
    def __init__(self) -> None:
        pass
    
    def extract_feature(self, music:Song, model, *args, **kwargs):
        """ Generic extract feature function, where the model only require the music file path.
        
            Args:
                music (Song): a valid Song class instance.
                model (function) : a function that given a music file path and posibly some other parameters returns some information about the song.
                args and kwargs : other parameteres of model.
            
            Returns:
                The extracted feature or features of \'music\' using \'model\'.
        """
        
        # TODO do some for of validation on music and model
        
        # TODO learn how to properly pass args and kwargs to  model
        return model(music.song_file, args, kwargs)

def downloaded_songs_names():
    downloaded_songs = {}
    BASE_DIR = Path(__file__).resolve().parent.parent
    # print(BASE_DIR)
    root = BASE_DIR.parent
    # print(root)
    music_folder = os.path.join(root,'download-musiccaps-dataset-main', 'music_data')
    # print(music_folder)
    files = os.listdir(music_folder)
    for file in files:
        # file = 'file_name + ext'        
        full_path = os.path.join(music_folder, file)
        name_only = os.path.splitext(file)[0]
        if '.part' in file:
            print("Skipping ",file)
            continue
        # print(name_only)
        downloaded_songs[name_only] = full_path
    # 3826 songs
    # print(len(downloaded_songs), "songs")
    return downloaded_songs

def musiccaps_preprocess():    
    """             ytid  start_s  end_s ... author_id  is_balanced_subset  is_audioset_eval
        0     -0Gj8-vB1q4     30     40  ...       4              False                 True
        1     -0SdAVK79lg     30     40  ...       0              False                False
        2     -0vPFx-wRRI     30     40  ...       6              False                 True
        3     -0xzrMun0Rs     30     40  ...       6              False                 True
        4     -1LrH01Ei1w     30     40  ...       0              False                False
        ...           ...    ...    ...  ...      ...               ...                  ... 
        5516  zw5dkiklbhE     15     25  ...       6              False                False 
        5517  zwfo7wnXdjs     30     40  ...       1               True                 True 
        5518  zx_vcwOsDO4     50     60  ...       2               True                 True 
        5519  zyXa2tdBTGc     30     40  ...       1              False                False 
        5520  zzNdwF40ID8     70     80  ...       9               True                 True 
        [5521 rows x 9 columns]"""
    musiccaps_csv_all_df = pd.read_csv(musiccaps_csv_all_path)
    # print(musiccaps_csv_all_df.to_string()) # prints the entire dataframe
    # print(musiccaps_csv_all_df)  # If you have a large DataFrame with many rows, Pandas will only return the first 5 rows, and the last 5 rows:  
    # subset_df = pd.DataFrame(musiccaps_csv_all_df, columns=["ytid", "author_id"]) # select a subset of columns from the CSV file
    # data_top = musiccaps_csv_all_df.head()
    
    columns_names = musiccaps_csv_all_df.columns.values # numpy.ndarray
    ## ['ytid' 'start_s' 'end_s' 'audioset_positive_labels' 'aspect_list' 'caption' 'author_id' 'is_balanced_subset' 'is_audioset_eval']
    subset1_by_cols = musiccaps_csv_all_df[["ytid", "start_s", "end_s","aspect_list", "caption"]]
    downloaded_songs = downloaded_songs_names()

    subset1_indexs = []
    for song in downloaded_songs:
        row = subset1_by_cols.loc[subset1_by_cols["ytid"] == song] # type: DataFrame
        if len(row.index) == 1:
            subset1_indexs.append(row.index[0])
        else: 
            print(row)
            raise Exception(f"There are more than one song with the same name {song}")
    
    subset1 = subset1_by_cols[subset1_by_cols.index.isin(subset1_indexs)]    
    subset1.to_csv(os.path.join(directory_path,'data','musiccaps-subset_index.csv')) # , header=False, index=False
    subset1.to_csv(os.path.join(directory_path,'data','musiccaps-subset.csv'), index=False) 
    # print(a_row.to_string())
    # print(subset1_by_cols[subset1_by_cols.index.isin([200,0])])

    return subset1

    




