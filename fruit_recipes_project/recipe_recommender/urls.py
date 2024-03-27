from django.urls import path
from . import views
from .views import register, index

app_name = 'recipe_recommender'  

urlpatterns = [
    path('update_preferences/', views.update_preferences, name='update_preferences'),
    path('upload/', views.upload_image, name='upload_image'),
    path('register/', register, name='register'),
    path('', index, name='index'),
]

