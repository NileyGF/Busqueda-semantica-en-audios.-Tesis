import src.BERT_embedding as BERT_embedding
import nltk
import pickle
import numpy as np
# nltk.download('punkt')

def evaluate_query(query_vector, documents_vectors_list,  top_k="all"):
    """
    Evaluates the query against the corpus 

    :returns: list of matching documents
    """
    doc_likehood = {}
    
    for k in range(len(documents_vectors_list)):
        cosine_sim = BERT_embedding.np_cosine_similarity(query_vector, documents_vectors_list[k])
        # cosine_sim = BERT_embedding.cosine_distance(query_vector, documents_vectors_list[k])

        doc_likehood[k] = cosine_sim # TODO k represents the index of the document in the django app
    
    ranked_docs = dict(sorted(doc_likehood.items(), key=lambda item: item[1], reverse=True))

    # Return the top_k docs with non-0-relevance
     
    i = 0
    while i < len(list(ranked_docs.values())) and list(ranked_docs.values())[i] > 1e-8:
        i += 1
        # if top_k != "all" and i >= top_k: 
        #     break
    # if isinstance(top_k, int) and top_k > 0 and i < top_k: 
    #     top_k = i
    
    if top_k == "all":
        if i < len(documents_vectors_list):
            return list(ranked_docs.keys())[0:i]
        return list(ranked_docs.keys())

    top_k = min(top_k, i)

    index_list = list(ranked_docs.keys())[0:top_k]
    
    return index_list

def extract_embeddings_for_docs_list(documents_list:list, save=False, save_path='corpus_bert_embeddings.bin', append=False):
    embeddings_list = []
    number_of_docs = len(documents_list)
    for i, text in enumerate(documents_list):
        # sent = nltk.sent_tokenize(text) # str list
        # print(sent)
        print(i, text)
        tokenized = BERT_embedding.bert_tokenize(text=text)
        print("tokens = ",len(tokenized))
        if len(tokenized) > 512:
            print(f"The BERT tokens, for the text number {i}, length is longer than the specified maximum sequence length . {len(tokenized)} > 512. ")
            print(text)
            raise Exception()
        embedding, _ = BERT_embedding.sentential_embeddings(tokenized_text=tokenized)
        print("embedding length =",len(embedding)) # 768
        embeddings_list.append(embedding)

    if save:
        if append:
            with open(save_path, 'rb') as f:
                old = pickle.load(f)
            append_embeddings_list = old + embeddings_list
            with open(save_path, 'wb') as f:
                pickle.dump(append_embeddings_list, f)
            return embeddings_list, append_embeddings_list
        # Save sentence embeddings
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_list, f)
    return embeddings_list

# extract_embeddings_for_docs_list(["The low quality recording features a ballad song that contains sustained strings, mellow piano melody and soft female vocal singing over it. It sounds sad and soulful, like something you would hear at Sunday services."])

def process_query(query:str, docs_embeddings_list:list=None, documents_list:list=None, top_k="all"):
    if docs_embeddings_list == None and documents_list == None:
        raise Exception("To process the query is necessary to have a corpus, either on a text list or an embeddings list.")
    if docs_embeddings_list == None and documents_list != None:
        docs_embeddings_list = extract_embeddings_for_docs_list(documents_list)
    
    tokenized_query = BERT_embedding.bert_tokenize(text=query)
    if len(tokenized_query) > 512:
        print(f"The BERT tokens, for the query, length is longer than the specified maximum sequence length . {len(tokenized_query)} > 512. ")
        print(query)
        raise Exception()
    query_embedding, _ = BERT_embedding.sentential_embeddings(tokenized_text=tokenized_query)
    
    relevant_docs_idx = evaluate_query(query_vector=query_embedding, documents_vectors_list=docs_embeddings_list, top_k=top_k)
    return relevant_docs_idx

