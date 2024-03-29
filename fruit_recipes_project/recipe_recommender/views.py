from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoModelForImageClassification, ViTImageProcessor
from django.shortcuts import render
from .forms import UploadPhotoForm, FeedbackForm, ProfileForm, CustomUserChangeForm
from .models import Profile
from PIL import Image
import torch
import numpy as np  # Ensure NumPy is imported

# Load the Hugging Face model and feature extractor
model_name = "PedroSampaio/fruits-360-16-7"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in
            return redirect('edit_profile')  
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def index(request):
    return render(request, 'index.html')

@login_required
def profile_view(request):
    return render(request, 'recipe_recommender/profile.html', {'user': request.user})

def home_view(request):
    return render(request, 'home.html')

@login_required
def edit_profile_view(request):
    try:
        profile = request.user.profile
    except Profile.DoesNotExist:
        profile = Profile(user=request.user)

    if request.method == 'POST':
        user_form = CustomUserChangeForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, request.FILES, instance=profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('profile')  # Make sure 'profile' is the name of the route to the user's profile
    else:
        user_form = CustomUserChangeForm(instance=request.user)
        profile_form = ProfileForm(instance=profile)

    context = {
        'user_form': user_form,
        'profile_form': profile_form
    }
    return render(request, 'recipe_recommender/edit_profile.html', context)

@csrf_exempt
def preprocess_image(image):
    # Preprocess the image here...
    image = image.resize((224, 224))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor /= 255.0
    return image_tensor

def postprocess_predictions(predictions):
    # Postprocess predictions here...
    predictions = predictions.argmax(dim=1)
    recognized_ingredients = [model.config.id2label[pred.item()] for pred in predictions]
    return recognized_ingredients

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file part'}, status=400)
        
        file = request.FILES['file']
        image = Image.open(file)
        image_tensor = preprocess_image(image)

        # Use the model and feature extractor for predictions
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            predictions = model(**inputs)

        recognized_ingredients = postprocess_predictions(predictions.logits)
        return JsonResponse({'ingredients': recognized_ingredients})

    else:
        # Handle non-POST requests or return a helpful error message
        return JsonResponse({'error': 'This endpoint only supports POST requests.'}, status=405)

def update_preferences(request):
    try:
        profile = request.user.profile
    except Profile.DoesNotExist:
        profile = Profile(user=request.user)
    
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': 'Preferences updated successfully'})
            else:
                # Redirect to a new URL, for example, the profile page
                return redirect('profile')  # Ensure you have a URL name 'profile' defined in your urls.py
    else:
        form = ProfileForm(instance=profile)
    
    return render(request, 'recipe_recommender/update_preferences.html', {'form': form})

def upload_photo(request):
    if request.method == 'POST':
        form = UploadPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            # Your logic here
            pass
    else:
        form = UploadPhotoForm()
    return render(request, 'upload_pic.html', {'form': form})

def feedbackform(request):
    if request.method == 'POST':
        fb_form = FeedbackForm(request.POST)
        if fb_form.is_valid():
            feedback = fb_form.save(commit=False)
            feedback.user = request.user
            feedback.save()
            # Process feedback data and update arrays here
            return redirect('upload_page')  
    else:
        fb_form = FeedbackForm()
    return render(request, 'feedback_form.html', {'form': fb_form})
