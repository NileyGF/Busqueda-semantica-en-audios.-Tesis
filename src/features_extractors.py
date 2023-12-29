from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

from ast import literal_eval
import numpy as np
import os

from src.core import Song, FeaturesExtractor, directory_path

essentia_models_path = os.path.join(directory_path, "essentia_models")

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
    classes = ["Blues---Boogie Woogie", "Blues---Chicago Blues", "Blues---Country Blues", "Blues---Delta Blues", "Blues---Electric Blues", "Blues---Harmonica Blues", "Blues---Jump Blues", "Blues---Louisiana Blues", "Blues---Modern Electric Blues", "Blues---Piano Blues", "Blues---Rhythm & Blues", "Blues---Texas Blues",
    "Brass & Military---Brass Band", "Brass & Military---Marches", "Brass & Military---Military", "Children's---Educational", "Children's---Nursery Rhymes", "Children's---Story",
    "Classical---Baroque", "Classical---Choral", "Classical---Classical", "Classical---Contemporary", "Classical---Impressionist", "Classical---Medieval", "Classical---Modern", "Classical---Neo-Classical", "Classical---Neo-Romantic", "Classical---Opera", "Classical---Post-Modern", "Classical---Renaissance", "Classical---Romantic",
    "Electronic---Abstract", "Electronic---Acid", "Electronic---Acid House", "Electronic---Acid Jazz", "Electronic---Ambient", "Electronic---Bassline", "Electronic---Beatdown", "Electronic---Berlin-School", "Electronic---Big Beat", "Electronic---Bleep", "Electronic---Breakbeat", "Electronic---Breakcore", "Electronic---Breaks", 
    "Electronic---Broken Beat", "Electronic---Chillwave", "Electronic---Chiptune", "Electronic---Dance-pop", "Electronic---Dark Ambient", "Electronic---Darkwave", "Electronic---Deep House", "Electronic---Deep Techno", "Electronic---Disco", "Electronic---Disco Polo", "Electronic---Donk", "Electronic---Downtempo", "Electronic---Drone", 
    "Electronic---Drum n Bass", "Electronic---Dub", "Electronic---Dub Techno", "Electronic---Dubstep", "Electronic---Dungeon Synth", "Electronic---EBM", "Electronic---Electro", "Electronic---Electro House", "Electronic---Electroclash", "Electronic---Euro House", "Electronic---Euro-Disco", "Electronic---Eurobeat", "Electronic---Eurodance", 
    "Electronic---Experimental", "Electronic---Freestyle", "Electronic---Future Jazz", "Electronic---Gabber", "Electronic---Garage House", "Electronic---Ghetto", "Electronic---Ghetto House", "Electronic---Glitch", "Electronic---Goa Trance", "Electronic---Grime", "Electronic---Halftime", "Electronic---Hands Up", "Electronic---Happy Hardcore", 
    "Electronic---Hard House", "Electronic---Hard Techno", "Electronic---Hard Trance", "Electronic---Hardcore", "Electronic---Hardstyle", "Electronic---Hi NRG", "Electronic---Hip Hop", "Electronic---Hip-House", "Electronic---House", "Electronic---IDM", "Electronic---Illbient", "Electronic---Industrial", "Electronic---Italo House", 
    "Electronic---Italo-Disco", "Electronic---Italodance", "Electronic---Jazzdance", "Electronic---Juke", "Electronic---Jumpstyle", "Electronic---Jungle", "Electronic---Latin", "Electronic---Leftfield", "Electronic---Makina", "Electronic---Minimal", "Electronic---Minimal Techno", "Electronic---Modern Classical", "Electronic---Musique Concr\u00e8te", 
    "Electronic---Neofolk", "Electronic---New Age", "Electronic---New Beat", "Electronic---New Wave", "Electronic---Noise", "Electronic---Nu-Disco", "Electronic---Power Electronics", "Electronic---Progressive Breaks", "Electronic---Progressive House", "Electronic---Progressive Trance", "Electronic---Psy-Trance", "Electronic---Rhythmic Noise", 
    "Electronic---Schranz", "Electronic---Sound Collage", "Electronic---Speed Garage", "Electronic---Speedcore", "Electronic---Synth-pop", "Electronic---Synthwave", "Electronic---Tech House", "Electronic---Tech Trance", "Electronic---Techno", "Electronic---Trance", "Electronic---Tribal", "Electronic---Tribal House", "Electronic---Trip Hop", "Electronic---Tropical House", "Electronic---UK Garage", "Electronic---Vaporwave",
    "Folk, World, & Country---African", "Folk, World, & Country---Bluegrass", "Folk, World, & Country---Cajun", "Folk, World, & Country---Canzone Napoletana", "Folk, World, & Country---Catalan Music", "Folk, World, & Country---Celtic", "Folk, World, & Country---Country", "Folk, World, & Country---Fado", "Folk, World, & Country---Flamenco", 
    "Folk, World, & Country---Folk", "Folk, World, & Country---Gospel", "Folk, World, & Country---Highlife", "Folk, World, & Country---Hillbilly", "Folk, World, & Country---Hindustani", "Folk, World, & Country---Honky Tonk", "Folk, World, & Country---Indian Classical", "Folk, World, & Country---La\u00efk\u00f3", "Folk, World, & Country---Nordic", 
    "Folk, World, & Country---Pacific", "Folk, World, & Country---Polka", "Folk, World, & Country---Ra\u00ef", "Folk, World, & Country---Romani", "Folk, World, & Country---Soukous", "Folk, World, & Country---S\u00e9ga", "Folk, World, & Country---Volksmusik", "Folk, World, & Country---Zouk", "Folk, World, & Country---\u00c9ntekhno",
    "Funk / Soul---Afrobeat", "Funk / Soul---Boogie", "Funk / Soul---Contemporary R&B", "Funk / Soul---Disco", "Funk / Soul---Free Funk", "Funk / Soul---Funk", "Funk / Soul---Gospel", "Funk / Soul---Neo Soul", "Funk / Soul---New Jack Swing", "Funk / Soul---P.Funk", "Funk / Soul---Psychedelic", "Funk / Soul---Rhythm & Blues", "Funk / Soul---Soul", "Funk / Soul---Swingbeat", "Funk / Soul---UK Street Soul",
    "Hip Hop---Bass Music", "Hip Hop---Boom Bap", "Hip Hop---Bounce", "Hip Hop---Britcore", "Hip Hop---Cloud Rap", "Hip Hop---Conscious", "Hip Hop---Crunk", "Hip Hop---Cut-up/DJ", "Hip Hop---DJ Battle Tool", "Hip Hop---Electro", "Hip Hop---G-Funk", "Hip Hop---Gangsta", "Hip Hop---Grime", "Hip Hop---Hardcore Hip-Hop", 
    "Hip Hop---Horrorcore", "Hip Hop---Instrumental", "Hip Hop---Jazzy Hip-Hop", "Hip Hop---Miami Bass", "Hip Hop---Pop Rap", "Hip Hop---Ragga HipHop", "Hip Hop---RnB/Swing", "Hip Hop---Screw", "Hip Hop---Thug Rap", "Hip Hop---Trap", "Hip Hop---Trip Hop", "Hip Hop---Turntablism",
    "Jazz---Afro-Cuban Jazz", "Jazz---Afrobeat", "Jazz---Avant-garde Jazz", "Jazz---Big Band", "Jazz---Bop", "Jazz---Bossa Nova", "Jazz---Contemporary Jazz", "Jazz---Cool Jazz", "Jazz---Dixieland", "Jazz---Easy Listening", "Jazz---Free Improvisation", "Jazz---Free Jazz", "Jazz---Fusion", "Jazz---Gypsy Jazz", 
    "Jazz---Hard Bop", "Jazz---Jazz-Funk", "Jazz---Jazz-Rock", "Jazz---Latin Jazz", "Jazz---Modal", "Jazz---Post Bop", "Jazz---Ragtime", "Jazz---Smooth Jazz", "Jazz---Soul-Jazz", "Jazz---Space-Age", "Jazz---Swing",
    "Latin---Afro-Cuban", "Latin---Bai\u00e3o", "Latin---Batucada", "Latin---Beguine", "Latin---Bolero", "Latin---Boogaloo", "Latin---Bossanova", "Latin---Cha-Cha", "Latin---Charanga", "Latin---Compas", "Latin---Cubano", "Latin---Cumbia", "Latin---Descarga", "Latin---Forr\u00f3", "Latin---Guaguanc\u00f3", "Latin---Guajira", 
    "Latin---Guaracha", "Latin---MPB", "Latin---Mambo", "Latin---Mariachi", "Latin---Merengue", "Latin---Norte\u00f1o", "Latin---Nueva Cancion", "Latin---Pachanga", "Latin---Porro", "Latin---Ranchera", "Latin---Reggaeton", "Latin---Rumba", "Latin---Salsa", "Latin---Samba", "Latin---Son", "Latin---Son Montuno", "Latin---Tango", "Latin---Tejano", "Latin---Vallenato",
    "Non-Music---Audiobook", "Non-Music---Comedy", "Non-Music---Dialogue", "Non-Music---Education", "Non-Music---Field Recording", "Non-Music---Interview", "Non-Music---Monolog", "Non-Music---Poetry", "Non-Music---Political", "Non-Music---Promotional", "Non-Music---Radioplay", "Non-Music---Religious", "Non-Music---Spoken Word",
    "Pop---Ballad", "Pop---Bollywood", "Pop---Bubblegum", "Pop---Chanson", "Pop---City Pop", "Pop---Europop", "Pop---Indie Pop", "Pop---J-pop", "Pop---K-pop", "Pop---Kay\u014dkyoku", "Pop---Light Music", "Pop---Music Hall", "Pop---Novelty", "Pop---Parody", "Pop---Schlager", "Pop---Vocal",
    "Reggae---Calypso", "Reggae---Dancehall", "Reggae---Dub", "Reggae---Lovers Rock", "Reggae---Ragga", "Reggae---Reggae", "Reggae---Reggae-Pop", "Reggae---Rocksteady", "Reggae---Roots Reggae", "Reggae---Ska", "Reggae---Soca",
    "Rock---AOR", "Rock---Acid Rock", "Rock---Acoustic", "Rock---Alternative Rock", "Rock---Arena Rock", "Rock---Art Rock", "Rock---Atmospheric Black Metal", "Rock---Avantgarde", "Rock---Beat", "Rock---Black Metal", "Rock---Blues Rock", "Rock---Brit Pop", "Rock---Classic Rock", "Rock---Coldwave", "Rock---Country Rock", 
    "Rock---Crust", "Rock---Death Metal", "Rock---Deathcore", "Rock---Deathrock", "Rock---Depressive Black Metal", "Rock---Doo Wop", "Rock---Doom Metal", "Rock---Dream Pop", "Rock---Emo", "Rock---Ethereal", "Rock---Experimental", "Rock---Folk Metal", "Rock---Folk Rock", "Rock---Funeral Doom Metal", "Rock---Funk Metal", 
    "Rock---Garage Rock", "Rock---Glam", "Rock---Goregrind", "Rock---Goth Rock", "Rock---Gothic Metal", "Rock---Grindcore", "Rock---Grunge", "Rock---Hard Rock", "Rock---Hardcore", "Rock---Heavy Metal", "Rock---Indie Rock", "Rock---Industrial", "Rock---Krautrock", "Rock---Lo-Fi", "Rock---Lounge", "Rock---Math Rock", 
    "Rock---Melodic Death Metal", "Rock---Melodic Hardcore", "Rock---Metalcore", "Rock---Mod", "Rock---Neofolk", "Rock---New Wave", "Rock---No Wave", "Rock---Noise", "Rock---Noisecore", "Rock---Nu Metal", "Rock---Oi", "Rock---Parody", "Rock---Pop Punk", "Rock---Pop Rock", "Rock---Pornogrind", "Rock---Post Rock", 
    "Rock---Post-Hardcore", "Rock---Post-Metal", "Rock---Post-Punk", "Rock---Power Metal", "Rock---Power Pop", "Rock---Power Violence", "Rock---Prog Rock", "Rock---Progressive Metal", "Rock---Psychedelic Rock", "Rock---Psychobilly", "Rock---Pub Rock", "Rock---Punk", "Rock---Rock & Roll", "Rock---Rockabilly", "Rock---Shoegaze", 
    "Rock---Ska", "Rock---Sludge Metal", "Rock---Soft Rock", "Rock---Southern Rock", "Rock---Space Rock", "Rock---Speed Metal", "Rock---Stoner Rock", "Rock---Surf", "Rock---Symphonic Rock", "Rock---Technical Death Metal", "Rock---Thrash", "Rock---Twist", "Rock---Viking Metal", "Rock---Y\u00e9-Y\u00e9",
    "Stage & Screen---Musical", "Stage & Screen---Score", "Stage & Screen---Soundtrack", "Stage & Screen---Theme"]
    def __init__(self) -> None:
        super().__init__()        
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "genre_discogs400-discogs-effnet-1.pb") 
        
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input="serving_default_model_Placeholder", output="PartitionedCall:0")
    
    def __str__(self=None) -> str:
        return "Discogs400 genre"
    
    def feature_description(feature:str):
        genre_split = feature.split('---')
        if len(genre_split) != 2:
            raise Exception("Discogs400 genre description error")
        return f"The music genre is considered to be {genre_split[0]}, {genre_split[1]}. "

    def extract_feature(self, music:Song):
        """ Essentia's Discogs400 Music Genre Classification
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)

        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)
        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
    
class MTG_Jamendo_GenreClass(BaseGenreClass): 
    """ Multi-label classification with the genre subset of MTG-Jamendo Dataset (87 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
    """
    classes = [ "60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", "blues", "bluesrock", "bossanova",
                "breakbeat", "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", 
                "darkwave", "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", "electronic", "electronica", "electropop", 
                "ethno", "eurodance", "experimental", "folk", "funk", "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", 
                "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval", "metal", "minimal", 
                "newage", "newwave", "orchestral", "pop", "popfolk", "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", 
                "rock", "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", "trance", "triphop", "world", "worldfusion" ]
        
    def __init__(self) -> None:
        super().__init__()      
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mtg_jamendo_genre-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Sigmoid" 

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
      
    def __str__(self=None) -> str:
        return "MTG Jamendo genre"      

    def feature_description(feature:str):
        return f"The music genre sounds like {feature}. "

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo Music Genre Classification
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
    
    
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
    classes =[ "danceable", "not_danceable"]
    def __init__(self) -> None:
        super().__init__()       
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "danceability-discogs-effnet-1.pb") 
             
        head_input_name = "model/Placeholder"
        head_output_name = "model/Softmax"

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")

        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
                    
    def __str__(self=None) -> str:
        return "danceability"

    def feature_description(feature:str):
        descript = "danceable" if feature == "danceable" else "not danceable"
        return f"It is {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music danceability (2 classes): danceable, not_danceable.
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)

        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
    
class Happy_DiscogsClass(BaseMoodClass):
    """ Music \"happiness\" (2 classes): happy, non_happy.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["happy", "non_happy" ]

    def __init__(self) -> None:
        super().__init__()        
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mood_happy-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Softmax" 

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
                    
    def __str__(self=None) -> str:
        return "happy"

    def feature_description(feature:str):
        descript = "happy" if feature == "happy" else "not happy"
        return f"Its melody is {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"happiness\" (2 classes): happy, non_happy.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
  
class Sad_DiscogsClass(BaseMoodClass):
    """ Music \"sadness\" (2 classes): sad, non_sad.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["non_sad", "sad"]

    def __init__(self) -> None:
        super().__init__()      
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mood_sad-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Softmax"
    
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
                    
    def __str__(self=None) -> str:
        return "sad"

    def feature_description(feature:str):
        descript = "sad" if feature == "sad" else "not sad"
        return f"Its melody is {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"sadness\" (2 classes): sad, non_sad.
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
    
class Relaxed_DiscogsClass(BaseMoodClass):
    """ Music \"relaxation\" (2 classes): relaxed, non_relaxed.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["non_relaxed", "relaxed"]

    def __init__(self) -> None:
        super().__init__()             
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mood_relaxed-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Softmax" 

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
                   
    def __str__(self=None) -> str:
        return "relaxed" 

    def feature_description(feature:str):
        descript = "relaxing" if feature == "relaxed" else "not relaxing"
        return f"Its sound is {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's Discogs Music \"relaxation\" (2 classes): relaxed, non_relaxed.
        
            Args:
                music (Song): a valid Song class instance.
        """
        
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[0]
  
class MoodTheme_MTGJamendoClass(BaseMoodClass):
    """ Multi-label classification with mood and theme subset of the MTG-Jamendo Dataset (56 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial", "cool", "corporate", 
               "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", "film", "fun", "funny", 
               "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", 
               "motivational", "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", 
               "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting" ]

    def __init__(self) -> None:
        super().__init__()         
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mtg_jamendo_moodtheme-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Sigmoid" 

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
           
    def __str__(self=None) -> str:
        return "mood and theme"

    def feature_description(feature:list):
        feature = literal_eval(feature)
        descript = ""
        for i in range(len(feature)-1):
            descript += f"{feature[i]}, "
        descript += f"{feature[len(feature)-1]}"

        return f"Some of the themes are {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo Music mood and theme.
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[:5] 
    
class Instruments_MTGJamendoClass(FeaturesExtractor):
    """ Multi-label classification using the instrument subset of the MTG-Jamendo Dataset (40 classes).
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", 
               "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", 
               "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", 
               "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"
               ]
    def __init__(self) -> None:
        super().__init__()          
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "mtg_jamendo_instrument-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Sigmoid" 

        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
      
    def __str__(self=None) -> str:
        return "instruments"

    def feature_description(feature:list):
        feature = literal_eval(feature)
        descript = ""
        for i in range(len(feature)-1):
            descript += f"{feature[i]}, "
        descript += f"{feature[len(feature)-1]}"

        return f"You can hear the sounds of {descript}. "

    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo instruments tags.
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        # return the class with the bigger value
        return list(sorted_data.keys())[:3] 
  
class VoiceGender_DiscogsClass(FeaturesExtractor):
    """ Classification of music by singing voice gender (2 classes): female, male.
    
        Essentia-tensorflow model

        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D 
    """
    classes = ["female", "male"]

    def __init__(self) -> None:
        super().__init__()           
        self.embedd_graph_file = os.path.join(essentia_models_path, "discogs-effnet-bs64-1.pb") 
        self.pred_graph_file = os.path.join(essentia_models_path, "gender-discogs-effnet-1.pb") 
        
        head_input_name = "model/Placeholder"
        head_output_name = "model/Softmax" 
        self.embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=self.embedd_graph_file, output="PartitionedCall:1")
        self.model = TensorflowPredict2D(graphFilename=self.pred_graph_file, input=head_input_name, output=head_output_name)
                    
    def __str__(self=None) -> str:
        return "voice gender"

    def feature_description(feature:str):
        return f"There is a {feature} voice. "
    
    def extract_feature(self, music:Song):
        """ Essentia's MTG-Jamendo instruments tags.
        
            Args:
                music (Song): a valid Song class instance.
        """
        # load audio
        audio = MonoLoader(filename=music.get_path(), sampleRate=16000, resampleQuality=4)()
        # get audio embeddings
        embeddings = self.embedding_model(audio)
        
        # tensorflow prediction
        predictions = self.model(embeddings)
        # tensorflow prediction values by classes
        predictions = np.mean(predictions, axis=0)

        # convert predictions in a dictionary with class_name:value
        pred_by_class = {}
        for i in range(len(self.classes)):
            pred_by_class[self.classes[i]] = predictions[i].astype(float)
        # TODO maybe the sorting is unnecesary
        # sort the dictionary in descending order by the value
        sorted_data = dict(sorted(list(pred_by_class.items()), key=lambda x: x[1], reverse=True))
        print("gender: ",sorted_data)
        # return the class with the bigger value
        return list(sorted_data.keys())[0] 
  

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