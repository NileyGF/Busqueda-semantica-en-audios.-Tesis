from django.db import models

class Music(models.Model):
    df_name = models.CharField(max_length=200, unique=True)
    description = models.CharField(max_length=500, default="Description ...")
    def __str__(self):
	    return self.df_name
