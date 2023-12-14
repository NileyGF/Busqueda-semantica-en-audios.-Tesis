from django.shortcuts import render
import os
from django_app.settings import BASE_DIR, CORE_PARENT_DIR
from .models import Music
import src.main as main

def home(request):
    main.save_embedd() # os.path.join(CORE_PARENT_DIR,'src','data','temp_corpus_bert_embeddings.bin')
    index = os.path.join(BASE_DIR,'retrieval','templates','index.html')
    all_songs = Music.objects.all()
    context = {'all_songs' : all_songs, 
               'color' : '#000080', 
               'top_k' : 'all', 
               'query' : None,
               'errors' : False,
               'results' : False}

    if request.method == 'POST' and request.POST.get('search_query',False) != False:
        top_k = request.POST['top_k']
        query:str = request.POST['query']
        context['query'] = query
        words = query.split(' ')
        if len(words) > 500:
            print("The query must have less than 500 words.")
            context['errors'] = "The query must have less than 500 words."
            return render(request, index, context)
        # print(top_k)
        if top_k != 'all':
            top_k = int(top_k)
        #     print(top_k)
        # print(query)
        context['top_k'] = top_k
        songs_idx = main.relevant_descriptions_by_query(query=query, top_k=top_k, embeddings_path=os.path.join(CORE_PARENT_DIR,'src','data','temp_corpus_bert_embeddings.bin'))
        # songs_idx = [1,0]
        # if top_k != 'all':
        #     songs_idx = songs_idx[:top_k]
        songs_list = []
        for idx in songs_idx:
            s = Music.objects.filter(index=idx)[0]
            # print(idx,s)
            songs_list.append(s)
        context['all_songs'] = songs_list
        context['results'] = True 
        
    # Music.objects.all().delete()
        
    # new = Music(df_name="-0Gj8-vB1q4", description="low quality, sustained strings melody, soft female vocal, mellow piano melody, sad, soulful, ballad",
    #              index=0, path=os.path.join("music_examples","-0Gj8-vB1q4.wav"))
    # new.save()
    # new = Music(df_name="-0SdAVK79lg", description="guitar song, piano backing, simple percussion, relaxing melody, slow tempo, bass, country feel, instrumental, no voice",
    #              index=1, path=os.path.join("music_examples","-0SdAVK79lg.wav"))
    # new.save()
    # print(context['results'])
    return render(request, index, context)
