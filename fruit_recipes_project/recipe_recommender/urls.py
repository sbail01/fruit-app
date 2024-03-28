from django.urls import path
from . import views
from .views import register, index, profile_view, upload_photo

app_name = 'recipe_recommender'  

urlpatterns = [
    path('update_preferences/', views.update_preferences, name='update_preferences'),
    path('pic-uploading-page/',upload_photo, name='upload_photo'),
    path('upload/', views.upload_image, name='upload_image'),
    path('register/', register, name='register'),
    path('', index, name='index'),
    path('accounts/profile/', profile_view, name='profile'),
]
