from django.urls import path, include, re_path
from test_group import views

urlpatterns = [
    path("", views.add_files, name='add_test_files'), # add-test-files/
    path("retrieval", views.retrieval, name='retrieval'),
]
# urlpatterns = [
#     re_path(r'^form/$', views.Form),
#     re_path(r'^upload/$', views.Upload)
# ]