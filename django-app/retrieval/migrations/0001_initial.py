# Generated by Django 4.2.3 on 2023-12-09 02:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Music',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('df_name', models.CharField(max_length=200, unique=True)),
            ],
        ),
    ]
