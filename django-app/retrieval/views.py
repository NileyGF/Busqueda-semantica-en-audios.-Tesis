from django.shortcuts import render
import os
from django_app.settings import BASE_DIR
from .models import Music

def home(request):
    index = os.path.join(BASE_DIR,'retrieval','templates','index.html')
    # new = Music(df_name = "-0Gj8-vB1q4",description="low quality, sustained strings melody, soft female vocal, mellow piano melody, sad, soulful, ballad")
    # new.save()
    # new = Music(df_name = "-0SdAVK79lg",description="guitar song, piano backing, simple percussion, relaxing melody, slow tempo, bass, country feel, instrumental, no voice")
    # new.save()
    all_songs = Music.objects.all()
    context = {'all_songs' : all_songs, 'color' : '#000080' }
    return render(request, index, context)
