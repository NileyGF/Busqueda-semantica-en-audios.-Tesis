# import essentia 
# import librosa
# from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

from src.core import Song, FeaturesExtractor


class BaseGenreClass(FeaturesExtractor):
    """ Base class for genre classifiers"""
    def __init__(self) -> None:
        super().__init__()
    
    def __str__(self=None) -> str:
        return "genre"

class Discogs400_GenreClass(BaseGenreClass):
    """ Music style classification by 400 styles from the Discogs taxonomy.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    
    def __str__(self=None) -> str:
        return "Discogs400 genre"

    def extract_feature(self, music:Song):
        """ Essentia's Discogs400 Music Genre Classification
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
    
class MTG_Jamendo_GenreClass(BaseGenreClass): 
    """ Multi-label classification with the genre subset of MTG-Jamendo Dataset (87 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mtg_jamendo_genre-discogs-effnet-1.pb")
      
    def __str__(self=None) -> str:
        return "MTG Jamendo genre"      

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo Music Genre Classification
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
    
class BaseMoodClass(FeaturesExtractor):
    """ Base class for mood and context classifiers"""
    def __init__(self) -> None:
        super().__init__()
            
    def __str__(self=None) -> str:
        return "mood"

class Danceability_DiscogsClass(BaseMoodClass):
    """ Music danceability (2 classes): danceable, not_danceable.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="danceability-discogs-effnet-1.pb", output="model/Softmax")
                    
    def __str__(self=None) -> str:
        return "danceability"

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music danceability (2 classes): danceable, not_danceable.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
    
class Happy_DiscogsClass(BaseMoodClass):
    """ Music \"happiness\" (2 classes): happy, non_happy.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mood_happy-discogs-effnet-1.pb", output="model/Softmax")
                    
    def __str__(self=None) -> str:
        return "happy"

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"happiness\" (2 classes): happy, non_happy.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
  
class Sad_DiscogsClass(BaseMoodClass):
    """ Music \"sadness\" (2 classes): sad, non_sad.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mood_sad-discogs-effnet-1.pb", output="model/Softmax")
                    
    def __str__(self=None) -> str:
        return "sad"

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"sadness\" (2 classes): sad, non_sad.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
    
class Relaxed_DiscogsClass(BaseMoodClass):
    """ Music \"relaxation\" (2 classes): relaxed, non_relaxed.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mood_relaxed-discogs-effnet-1.pb", output="model/Softmax")
                   
    def __str__(self=None) -> str:
        return "relaxed" 

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"relaxation\" (2 classes): relaxed, non_relaxed.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
  
class MoodTheme_MTGJamendoClass(BaseMoodClass):
    """ Multi-label classification with mood and theme subset of the MTG-Jamendo Dataset (56 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
                    
    def __str__(self=None) -> str:
        return "mood and theme"

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo Music mood and theme.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
    
class Instruments_MTGJamendoClass(FeaturesExtractor):
    """ Multi-label classification using the instrument subset of the MTG-Jamendo Dataset (40 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
                    
    def __str__(self=None) -> str:
        return "instruments"

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo instruments tags.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
  
class VoiceGender_DiscogsClass(FeaturesExtractor):
    """ Classification of music by singing voice gender (2 classes): female, male.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    def __init__(self) -> None:
        super().__init__()        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename="gender-discogs-effnet-1.pb", output="model/Softmax")
                    
    def __str__(self=None) -> str:
        return "voice gender"

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo instruments tags.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        audio = MonoLoader(filename=music.get_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.embedding_model(audio)

        predictions = self.model(embeddings)

        return predictions
  

SPECIFIC_FEATURES = {
    Discogs400_GenreClass.__str__(): Discogs400_GenreClass,
    MTG_Jamendo_GenreClass.__str__(): MTG_Jamendo_GenreClass,
    Danceability_DiscogsClass.__str__(): Danceability_DiscogsClass,
    Happy_DiscogsClass.__str__(): Happy_DiscogsClass,
    Sad_DiscogsClass.__str__(): Sad_DiscogsClass,
    Relaxed_DiscogsClass.__str__(): Relaxed_DiscogsClass,
    MoodTheme_MTGJamendoClass.__str__(): MoodTheme_MTGJamendoClass,
    Instruments_MTGJamendoClass.__str__(): Instruments_MTGJamendoClass,
    VoiceGender_DiscogsClass.__str__(): VoiceGender_DiscogsClass,
}