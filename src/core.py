# Huggingface token : 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
import pandas as pd
import os

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
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
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path).parent.parent
    music_folder = os.path.join(directory_path, 'music_examples')
    song = os.path.join(music_folder,os.listdir(music_folder)[0])



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
    subset1 =  musiccaps_csv_all_df[["ytid", "start_s", "end_s","aspect_list", "caption"]]
    
    
    




