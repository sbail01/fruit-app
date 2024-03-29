from django.urls import path
from . import views
from .views import register, index, profile_view, upload_photo

app_name = 'recipe_recommender'  

urlpatterns = [
    path('update_preferences/', views.update_preferences, name='update_preferences'),
    path('pic-uploading-page/',upload_photo, name='upload_photo'),
    path('upload/', views.upload_image, name='upload_image'),
    path('register/', register, name='register'),
    path('', views.home_view, name='home'),
    path('accounts/profile/', profile_view, name='profile'),
    path('edit-profile/', views.edit_profile_view, name='edit_profile'),
    path('home/', views.home_view, name='home'),
    path('feedback/', views.feedbackform, name='feedbackform'),
]
