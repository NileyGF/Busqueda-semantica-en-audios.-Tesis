from django import forms
from .models import UploadMusic	

class MusicForm(forms.ModelForm):
    music = forms.FileField(widget=forms.TextInput(attrs={
            "name": "audios",
            "type": "File",
            "class": "form-control",
            "multiple": "True",
        }), label = "")
    class Meta:
        model = UploadMusic
        fields = ['music']

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=100)
    file = forms.FileField()

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result

class FileFieldForm(forms.Form):
    file_field = MultipleFileField()