from django import forms
from .models import Feedback, Profile
from django.contrib.auth.models import User

class UploadPhotoForm(forms.Form):
    photo = forms.ImageField()

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['fruit_or_vegetable', 'feedback_type', 'description']

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['age', 'allergies', 'dietary_restrictions', 'favourite_cuisine']

class CustomUserChangeForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')