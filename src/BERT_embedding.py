# # Attempt #1
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    sentence_1, vs_sum_cat1 = sentential_embeddings(tokenized_text_1)
    sentence_2, vs_sum_cat2 = sentential_embeddings(tokenizer,tokenized_text_2)
    # return vector_similiarity(sentence_1, sentence_2)
    # return cosine_distance(sentence_1, sentence_2)
    return np_cosine_similarity(sentence_1, sentence_2)

def cosine_distance(embedd1, embedd2):
    distance = 1 - cosine(embedd1, embedd2)
    return distance

def vector_similiarity(bert_vec1, bert_vec2):
    print(bert_vec1.size())
    print(bert_vec2.size())
    return cosine_similarity(bert_vec1, bert_vec2)

def np_cosine_similarity(bert_vec1, bert_vec2):
    A = np.array(bert_vec1)
    B = np.array(bert_vec2)
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    if A_norm == 0 or B_norm == 0: return 0

    cosine = np.dot(A,B) / ( A_norm * B_norm)
    return cosine
# f4='The Pacific Ocean is deeper than the Atlantic Ocean'
# f5='The Atlantic Ocean is smaller than the Pacific Ocean'

# dist=calculate_distance(f5,f4)
# print(dist)