import os
import pandas as pd
from django_app.settings import BASE_DIR, CORE_PARENT_DIR
from retrieval.models import Music
import src.main as main


# features_extracted_csv = os.path.join(main.directory_path,'data','musiccaps-subset-feat.csv')
descriptions_csv = os.path.join(main.directory_path,'data','musiccaps-subset-descriptions.csv')
# descriptions_embeddings_path = os.path.join(main.directory_path,'data','embeddings','corpus_bert_embeddings.bin')
# descriptions_extended_embeddings_path = os.path.join(main.directory_path,'data','embeddings','extended_corpus_bert_embeddings.bin')
queries_embeddings_path = os.path.join(main.directory_path,'data','embeddings','queries_bert_embeddings.bin')
# queries_relevance_path = os.path.join(main.directory_path,'data','queries_songs_relevance.bin')
queries2_embeddings_path = os.path.join(main.directory_path,'data','embeddings','queries2_bert_embeddings.bin')
# queries2_relevance_path = os.path.join(main.directory_path,'data','queries2_songs_relevance.bin')


def load_to_database(music_folder_path:str=None,append=True):
    if not append:
        Music.objects.all().delete()
    songs_path_dict = main.downloaded_songs_name_path(music_folder_path)
    
    descriptions_df = pd.read_csv(descriptions_csv)    
    # descriptions_list = descriptions_df["description"].values.tolist()
    for idx, row in descriptions_df.iterrows():
        print(idx) # TODO
        df_name = row['ytid']
        description = row['description']
        index = idx
        path = songs_path_dict[df_name]
        repeated = Music.objects.filter(df_name=df_name)
        if len(repeated) > 0:
            repeated.update(description=description,index=index,path=path)
        else:
            new = Music(df_name=df_name,description=description,index=index,path=path)
            new.save()
        
