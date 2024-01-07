from django.shortcuts import render, redirect
# from django.http import HttpResponse
# from django.views.generic.edit import FormView
from .forms import  MusicForm
from .models import UploadMusic
from .functions import process_test_group
import os
# from .load_models import load_to_database
import src.main as main
from django_app.settings import BASE_DIR, CORE_PARENT_DIR

upload = False

def retrieval(request):
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

def add_files(request):    
    global upload
    context = {'upload' : upload}
    
    upl_music = UploadMusic.objects.all()
    # upl_music.delete()
    context["uploaded"] =  upl_music

    form = MusicForm(request.POST, request.FILES)
    if request.method == "POST" and request.POST.get('upload_btn',False) != False:
        files = request.FILES.getlist('music')
        for f in files:
            # print(f)
            m = UploadMusic(file=f,name=f'{f}')
            m.save()
        context["upload"] = False
        upload = context['upload']
        # return render(request, "test_group.html", context)
        # if form.is_valid():
        #     handle_uploaded_file(request.FILES["file"])
        #     # return HttpResponseRedirect("/success/url/")
    if request.method == "POST" and request.POST.get('switch_btn',False) != False:
        
        if request.POST.get("upload_switch") == 'on':
            context['upload'] = True
        else: context['upload'] = False
        upload = context['upload']
    if request.method == "POST" and request.POST.get('process',False) != False:
        process_test_group()
        # redirect('retrieval')
    context["form"] = form
    return render(request, "test_group.html", context)
