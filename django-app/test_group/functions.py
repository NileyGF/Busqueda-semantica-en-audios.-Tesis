import os
import pandas as pd
import time
import pickle
import src.main as main
import src.core as core
from .models import UploadMusic
from django_app.settings import BASE_DIR, CORE_PARENT_DIR
data_dir = os.path.join(BASE_DIR, 'test_group', 'data')
music_dir = os.path.join(BASE_DIR, 'static', 'test')
descriptions_embeddings_path = os.path.join(data_dir,'corpus_bert_embeddings.bin')
relation_path = os.path.join(data_dir,'embeddings_rel.bin')

def process_test_group():
    # upl_music[0].file -> static/test/actual_file_name.mp3
    # upl_music[0].name -> original file name
    # upl_files_list[0] -> actual_file_name.mp3
    upl_music:list = UploadMusic.objects.all()
    # print([m.file for m in upl_music])
    upl_files_list = os.listdir(music_dir) 
    upl_files_list = core.downloaded_songs_name_path(music_dir)
    upl_files_path_dict = {}
    for file_name in upl_files_list:        
        new = UploadMusic.objects.filter(file__startswith=f'static/test/{file_name}')[0]
        print(new)
        name_only = os.path.splitext(new.name)[0]
        # print(new)
        upl_files_path_dict[name_only] = upl_files_list[file_name]
    df_d = {'name':list(upl_files_path_dict.keys())}
    first_df = pd.DataFrame.from_dict(df_d)
    first_df.to_csv(os.path.join(data_dir,'test-group-feat.csv'), index=False) 
    # print(first_df)  
    # First part of the pipeline
    features_df = extract_features(first_df, upl_files_path_dict)
    # Second part of the pipeline
    descriptions_df = get_descriptions_from_feat(features_df)
    # Third part of the pipeline
    save_descriptions_embedd(descriptions_df)

def extract_features(data_frame:pd.DataFrame, songs_dict:dict):
    features_dict = {'name': data_frame['name'].values.tolist()}
    for feat in main.SPECIFIC_FEATURES:
        features_dict[feat] = []
    st = time.time()
    for song in features_dict['name']:
        print("Extracting features from ",songs_dict[song])
        try:
            song_feat = main.extract_features(songs_dict[song])            
        except Exception as err:
            print("Error while extracting features: ", err)
            song_feat = {feat:'' for feat in main.SPECIFIC_FEATURES}
        for f in song_feat:
            if song_feat[f] == None:
                features_dict[f].append("")
            else:
                features_dict[f].append(song_feat[f])
    print(features_dict)
    print("Extract features time", round(time.time()-st,4)," sec")
    features_df = pd.DataFrame.from_dict(features_dict)

    inner_merged = pd.merge(data_frame, features_df)

    inner_merged.to_csv(os.path.join(data_dir,'test-group-feat.csv'), index=False) 
    return inner_merged    

def get_descriptions_from_feat(feat_df):
    st = time.time()
    features_dict = main.SPECIFIC_FEATURES
    description_dict = {'name':feat_df['name'].values.tolist(), 
                        'description':[], 'tags_description':[]}

    for idx, row in feat_df.iterrows():
        song_description = ""
        song_tag_description = ""
        print(idx)
        for feat in features_dict:
            extractor= features_dict[feat]
            feature_value = row[feat]
            song_description += extractor.feature_description(feature=feature_value)
            song_tag_description += extractor.feature_tags_description(feature=feature_value) + '; '
        song_tag_description = song_tag_description[:-2] + '.'
        description_dict['description'].append(song_description)
        description_dict['tags_description'].append(song_tag_description)
    
    print("Descriptions time", round(time.time()-st,4)," sec")
    descr_df = pd.DataFrame.from_dict(description_dict)
    inner_merged = pd.merge(feat_df, descr_df)
    
    inner_merged.to_csv(os.path.join(data_dir,'test-group-descriptions.csv'), index=False) 
    return inner_merged

def save_descriptions_embedd(descr_df):
    descriptions_list = descr_df["description"].values.tolist()
    tags_descriptions_list = descr_df["tags_description"].values.tolist()
    st = time.time()
    main.emb_ret.extract_embeddings_for_docs_list(documents_list=descriptions_list, save=True, save_path=descriptions_embeddings_path)
    with open(descriptions_embeddings_path, 'rb') as f:
        docs_embeddings_list = pickle.load(f)

    embedd_corpus_relation = [(i,i) for i in range(len(docs_embeddings_list))]
    with open(relation_path, 'wb') as f:
        pickle.dump(embedd_corpus_relation, f)

    print("Embeddings time", round(time.time()-st,4)," sec")
    return docs_embeddings_list
    



