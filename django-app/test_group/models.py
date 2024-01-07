from django.db import models
import os

class UploadMusic(models.Model):
    file = models.FileField(upload_to=os.path.join('static','test'))
    name = models.CharField(max_length=200, default="")
    # df_name = models.CharField(max_length=200, unique=True)
    # description = models.CharField(max_length=500, default="Description ...")
    # index = models.IntegerField(default=-1)#(unique=True)
    # path = models.TextField(default="/")
    # def __str__(self):
	#     return f"{self.index} - {self.df_name}"