
class Song:
    def __init__(self, file_path:str) -> None:
        self.song_file = file_path

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