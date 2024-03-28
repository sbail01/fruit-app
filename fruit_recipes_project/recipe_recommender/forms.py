from django import forms

class UploadPhotoForm(forms.Form):
    photo = forms.ImageField()
