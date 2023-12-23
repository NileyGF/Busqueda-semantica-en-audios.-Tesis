from django.shortcuts import render
import os
from django_app.settings import BASE_DIR, CORE_PARENT_DIR
from .models import Music
from .load_models import load_to_database
import src.main as main

descriptions_embeddings_path = os.path.join(CORE_PARENT_DIR,'src','data','embeddings','corpus_bert_embeddings.bin')
descriptions_extended_embeddings_path = os.path.join(CORE_PARENT_DIR,'src','data','embeddings','extended_corpus_bert_embeddings.bin')

def home(request):
    # print('hi')
    # main.save_embedd() # os.path.join(CORE_PARENT_DIR,'src','data','temp_corpus_bert_embeddings.bin')
    index = os.path.join(BASE_DIR,'retrieval','templates','index.html')
    all_songs = Music.objects.all()
    context = {'all_songs' : all_songs, 
               'color' : '#000080', 
               'top_k' : '50', 
               'query' : None,
               'errors' : False,
               'results' : False}
    print(request.POST)
    if request.method == 'POST' and request.POST.get('search_query',False) != False:
        top_k = request.POST['top_k']
        query:str = request.POST['query']
        context['query'] = query
        words = query.split(' ')
        if len(words) > 500:
            print("The query must have less than 500 words.")
            context['errors'] = "The query must have less than 500 words."
            return render(request, index, context)
        
        if top_k != 'all':
            top_k = int(top_k)
        context['top_k'] = top_k
        print('searching:')
        songs_idx = main.relevant_descriptions_by_query(query=query, top_k=top_k, embeddings_path=descriptions_embeddings_path)
        
        print('done')
        # songs_idx = [1,0]
        # if top_k != 'all':
        #     songs_idx = songs_idx[:top_k]
        # songs_list = Music.objects.filter(id__in=songs_idx)
        songs_list = []
        for idx in songs_idx:
            s = Music.objects.filter(index=idx)[0]
            # print(idx,s)
            songs_list.append(s)
        context['all_songs'] = songs_list
        context['results'] = True 
        
    if request.method == 'POST' and request.POST.get('load_musics',False) != False:
        load_to_database(append=True)
        all_songs = Music.objects.all()
        context['all_songs'] = all_songs

    return render(request, index, context)
