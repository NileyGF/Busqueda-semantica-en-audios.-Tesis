# Generated by Django 4.2.3 on 2023-12-12 00:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('retrieval', '0002_music_description'),
    ]

    operations = [
        migrations.AddField(
            model_name='music',
            name='index',
            field=models.IntegerField(default=-1),
        ),
        migrations.AddField(
            model_name='music',
            name='path',
            field=models.TextField(default='/'),
        ),
    ]
