# # Attempt #1
# import numpy as np
# import pandas as pd
# import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# def bert_vectorize(sentence):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     marked_text = "[CLS] " + sentence + " [SEP]"
#     print(marked_text)
    
#     tokenized_text = tokenizer.tokenize(marked_text)
#     print(tokenized_text)

#     segments_ids = [1] * len(tokenized_text)
#     print(segments_ids)

#     segments_tensors = torch.tensor([segments_ids])

#     indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#     tokens_tensor = torch.tensor([indexed_tokens])
#     print(tokens_tensor)

#     model = BertModel.from_pretrained('bert-base-uncased') # , output_hidden_states=True
    
#     with torch.no_grad():
#         encoded_layers, _ = model(tokens_tensor, segments_tensors)

#     sentence_embedding = torch.mean(encoded_layers[11], 1)
#     print(sentence_embedding)

#     return sentence_embedding

# def sentence_similiarity(text1, text2):
#     vec1 = bert_vectorize(text1)
#     vec2 = bert_vectorize(text2)
#     print(vec1.size)
#     print(vec1.shape)
#     return cosine_similarity(vec1, vec2)

def vector_similiarity(bert_vec1, bert_vec2):
    print(bert_vec1.size)
    print(bert_vec2.shape)
    return cosine_similarity(bert_vec1, bert_vec2)

# def normalize(df, feature_names):
#     result = df.copy()
#     for feature_name in feature_names:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result

# #text = ["I am unable to create new user", "I cannot create new user"]
# #text = ["What's in a name? That which we call a rose" ,"by any other name would smell as sweet"]
# #text = ["An example of pragmatics is how the same word can have different meanings in different settings.", "An example of pragmatics is the study of how people react to different symbols."]
# #text = ["Here is the sentence I want embeddings for.", "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."]
# #text = ["the branch of linguistics concerned with meaning in context, or the meanings of sentences in terms of the speaker's intentions in using them",
# #"the branch of semiotics dealing with the relationships of signs and symbols to their users"]
# text = ["I am good.", "I am not bad."]
# vec1 = bert_vectorize(text[0])
# vec2 = bert_vectorize(text[1])
# print(vector_similiarity(vec1, vec2))

# bert_similarity = []
# data = pd.read_csv('bert_test.csv')
# # data = normalize(data, ['sim'])

# for text1, text2 in text:
#     vec1 = bert_vectorize(text1)
#     vec2 = bert_vectorize(text2)

#     similarity = vector_similiarity(vec1, vec2)
#     bert_similarity.append(similarity)
#     series = pd.Series('bert_similarity')
#     data['bert_similarity'] = series.values

# Attempt #2
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_tokenize(text):
    preprocess_tokens = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(preprocess_tokens)
    return tokenized_text

def sentential_embeddings(tokenized_text):
    segmenter_idx = [1] * len(tokenized_text)
    segmenter_tensor = torch.tensor([segmenter_idx])

    idx_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([idx_tokens])
    
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segmenter_tensor)
        hidden_state = outputs[2]
    embedding_token = torch.stack(hidden_state, dim=0)
    embedding_token = torch.squeeze(embedding_token, dim=1)
    embedding_token = embedding_token.permute(1,0,2)
    vs_sum_cat = []
    for i in embedding_token:
        vs_li = torch.sum(i[-4:], dim=0)
        vs_sum_cat.append(vs_li)
    token_vecs = hidden_state[-2][0]
    sentence_embeddings = torch.mean(token_vecs, dim=0)
    return sentence_embeddings, vs_sum_cat

def calculate_distance(sentence_1,sentence_2):
    tokenized_text_1 = bert_tokenize(sentence_1)
    tokenized_text_2 = bert_tokenize(sentence_2)
    sentence_1, vs_sum_cat1 = sentential_embeddings(tokenizer,tokenized_text_1)
    sentence_2, vs_sum_cat2 = sentential_embeddings(tokenizer,tokenized_text_2)
    return vector_similiarity(sentence_1, sentence_2)
    return cosine_distance(sentence_1, sentence_2)

def cosine_distance(embedd1, embedd2):
    distance = 1 - cosine(embedd1, embedd2)
    return distance

# f4='The Pacific Ocean is deeper than the Atlantic Ocean'
# f5='The Atlantic Ocean is smaller than the Pacific Ocean'

# dist=calculate_distance(f5,f4)
# print(dist)