from django.urls import path
from . import views
from .views import register, index, profile_view

app_name = 'recipe_recommender'  

urlpatterns = [
    path('update_preferences/', views.update_preferences, name='update_preferences'),
    path('upload/', views.upload_image, name='upload_image'),
    path('register/', register, name='register'),
    path('', index, name='index'),
    path('accounts/profile/', profile_view, name='profile'),
]
