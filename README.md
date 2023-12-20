# Busqueda-semantica-en-audios.-Tesis

The current state of the tesis can be found [here](https://github.com/NileyGF/Busqueda-semantica-en-audios.-Tesis/blob/main/docs/Recuperaci%C3%B3n_sem%C3%A1ntica_de_m%C3%BAsica_utilizando_grandes_modelos_de_lenguaje_y_embeddings.pdf).

Besides de documents on the repository, the analysis of the state of the art on Music Information Retrieval(MIR), is on the google sheet [here](https://docs.google.com/spreadsheets/d/1_MJO6jbfSJLG0gLDlu911yQ0zc3quQeut_394DEvVk4/edit?usp=sharing).

Machine learning, Information Retrieval Systems, Music Information Retrieval, BERT, essentia, Music Classification, NLP

source env/bin/activate # ubuntu linux

Set-ExecutionPolicy Unrestricted -Scope Process # windows
env\bin\Activate.ps1

conda activate ess

python3 -m src.test.py

cd django-app

python3 manage.py runserver