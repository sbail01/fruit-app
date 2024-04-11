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
from django.conf import settings
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from django.core.files.storage import default_storage
from keras.preprocessing.image import img_to_array, load_img
import torch
import numpy as np  # Ensure NumPy is imported
import os
from torchvision import transforms
from roboflow import Roboflow
import supervision as sv
import csv
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import openai

# # Load the Hugging Face model and feature extractor
# model_name = "PedroSampaio/fruits-360-16-7"

# model = AutoModelForImageClassification.from_pretrained(model_name)
# feature_extractor = ViTImageProcessor.from_pretrained(model_name)

rf = Roboflow(api_key="iqS3KNWPgcu9iiGVHlLw")
project = rf.workspace().project("fruits-and-vegetables-2vf7u")
model = project.version(1).model
openai.api_key = "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"

# MODEL_PATH = os.path.join(settings.BASE_DIR, 'fruit_cnn_model.h5')
# fruit_model = load_model('C:/Users/Sabrina/Desktop/fruit-app/fruit_recipes_project/fruit_cnn_model.h5')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in
            return redirect('edit_profile')  # Redirect to the edit profile page
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
def apply_nms(detections, threshold=0.5):
    """
    Implements a simplified Non-Maximum Suppression algorithm.

    Args:
        detections (list): List of tuples (label, confidence, x, y) representing detected objects.
        threshold (float): The IoU (Intersection over Union) threshold for suppression.

    Returns:
        list: List of filtered detections.
    """

    # Sort detections by decreasing confidence score
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    filtered_detections = []
    while detections:
        # Select the detection with the highest confidence
        highest_confidence = detections.pop(0)
        filtered_detections.append(highest_confidence)

        # Calculate IoU between the highest confidence detection and the rest
        to_remove = []
        for i, det in enumerate(detections):
            iou = calculate_iou(highest_confidence[2:], det[2:])  # Assumes x, y are at index 2 and 3
            if iou > threshold:
                to_remove.append(i)

        # Remove suppressed detections
        for i in reversed(to_remove):
            detections.pop(i)

    return filtered_detections

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): Coordinates (x1, y1, x2, y2) of the first bounding box.
        box2 (tuple): Coordinates (x1, y1, x2, y2) of the second bounding box.

    Returns:
        float: IoU value between the two bounding boxes.
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection and union areas
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / float(union_area)
@csrf_exempt
# Define the preprocess function
def preprocess_image(image_path, target_size=(100, 100)):  # Verify the target_size used during training
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the preprocess to the image
    image_tensor = preprocess(image)
    # Add a batch dimension since models expect it
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def postprocess_predictions(logits):
    """Converts model logits to predictions with confidence scores."""
    probs = torch.nn.functional.softmax(logits, dim=1)
    top_probs, top_lbl_indices = probs.max(dim=1)
    labels = [model.config.id2label[idx.item()] for idx in top_lbl_indices]
    confidences = top_probs.tolist()  # Convert to list of confidence scores
    print("Predictions:", postprocess_predictions)
    print("Predicted class indices:", labels)

    return list(zip(labels, confidences))


def generate_sliding_windows(image, window_size, stride):
    """
    Generates overlapping windows for the sliding window approach and calculates their offsets.

    Args:
        image (torch.Tensor): The preprocessed image tensor.
        window_size (int): The width and height of the sliding window.
        stride (int): The overlapping step size between windows.

    Yields:
        tuple: (torch.Tensor, int, int) Individual image window tensors, and their x and y offsets.
    """

    # Extract image dimensions
    _, C, H, W = image.shape  

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            window = image[:, :, y:y + window_size, x:x + window_size]
            yield window, x, y  # Return the window and its offset
def parse_recipes(response):
    recipes = []
    recipe_texts = response.split("Name:")
    for recipe_text in recipe_texts[1:]:
        recipe_lines = recipe_text.split("\n")
        name = recipe_lines[0].strip()
        duration = int(recipe_lines[1].split()[1])
        steps = "\n".join(recipe_lines[3:]).strip()
        recipes.append({"Name": name, "Duration": duration, "Steps": steps})
    return recipes

# Function to generate recipes
def generate_recipes(ingredients):
    prompt = f"Ingredients: {', '.join(ingredients)}. Please generate three possible recipes. Only need recipe name (as Name:), duration (as Duration:) and steps (as Steps:). Do not need ingredients list again."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1500
    )

    return parse_recipes(response.choices[0].text.strip())

@csrf_exempt
def save_csv(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "This endpoint only accepts POST requests."}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        image_path = data.get('image_path', '')
        detected_items = data.get('detected_items', [])
        recipes = generate_recipes(detected_items)
        print(recipes[0]["Name"])  # Access the name of the first recipe
        print(recipes[1]["Duration"])  # Access the duration of the second recipe
        print(recipes[2]["Steps"])  # Access the steps of the third recipe
        # Check and prepare the CSV file path
        csv_file_path = os.path.join(settings.BASE_DIR, 'detected_items.csv')
        write_header = not os.path.exists(csv_file_path)
        
        # Prepare to manage the oldest entry
        rows = []
        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                rows = list(csv.reader(file))
            
            if len(rows) > 2:  # Modify this number based on your specific needs
                oldest_image_path = os.path.join(settings.BASE_DIR, rows[1][0])
                try:
                    os.remove(oldest_image_path)
                    print(f"Deleted oldest image: {oldest_image_path}")
                except Exception as e:
                    print(f"Error deleting the image {oldest_image_path}: {e}")
                rows.pop(1)  # Remove the oldest entry
                
            # Rewrite the CSV without the oldest entry
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
                csv.writer(file).writerows(rows)

        # Append new data to the CSV
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if write_header:
                header = ['Image Path'] + ['Item {}'.format(i + 1) for i in range(len(detected_items))]
                writer.writerow(header)
            row = [image_path] + detected_items
            writer.writerow(row)
        
        return JsonResponse({"success": True, "message": "Data successfully added to CSV."})
    except Exception as e:
        print(f"Error processing the request: {e}")
        return JsonResponse({"success": False, "message": "Error processing the request."}, status=400)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file part'}, status=400)

        file = request.FILES['file']
        image = Image.open(file).convert('RGB')

        # Define the uploads directory path
        uploads_dir = os.path.join(settings.BASE_DIR, 'uploads')
        # Check if the uploads directory exists, create it if it doesn't
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # # Define the full path for saving the image
        # save_path = os.path.join(uploads_dir, file.name)
        # if os.path.exists(save_path):
        #     base, extension = os.path.splitext(file.name)
        #     counter = 1
        #     while os.path.exists(os.path.join(uploads_dir, f"{base}_{counter}{extension}")):
        #         counter += 1
        #     new_filename = f"{base}_{counter}{extension}"
        #     save_path = os.path.join(uploads_dir, new_filename)
                # Append a timestamp to the file name
        base, extension = os.path.splitext(file.name)
        timestamp = timezone.now().strftime('%Y%m%d%H%M%S')  
        new_filename = f"{base}_{timestamp}{extension}"
        save_path = os.path.join(uploads_dir, new_filename)
        if not os.path.exists(save_path):
            image.save(save_path)

        # Predict using the saved image
        result = model.predict(save_path, confidence=30, overlap=40)

        # Extract the labels and confidence from the predictions
        # The previous line had a mistake by trying to access predictions as if they were attributes of an object.
        # We're now accessing them as keys in a dictionary.
        predictions = result.predictions if hasattr(result, 'predictions') else []
        print(predictions)
        detected_items = [{
            "class": prediction["class"],  # Corrected attribute access
            "confidence": prediction["confidence"],
            "bbox": {
                "x": prediction["x"],
                "y": prediction["y"],
                "width": prediction["width"],
                "height": prediction["height"]
            }
        } for prediction in predictions]
        # # Try to delete the saved image, catch and log any errors
        # try:
        #     os.remove(save_path)
        # except Exception as e:
        #     print(f"Error deleting the image {save_path}: {e}")

        # Return the predictions as JSON
        return JsonResponse({'detected_items': detected_items,'save_path':save_path})

    else:
        return JsonResponse({'error': 'This endpoint only supports POST requests.'}, status=405)




# def save_uploaded_file(uploaded_file):
#     """
#     Saves an uploaded file to the 'uploads' directory and returns the path.

#     Args:
#         uploaded_file (InMemoryUploadedFile): The file uploaded by the user.

#     Returns:
#         str: The file system path to the saved file.
#     """

#     # Define the uploads directory path
#     uploads_dir = os.path.join(settings.BASE_DIR, 'uploads')
#     # Ensure the uploads directory exists
#     if not os.path.exists(uploads_dir):
#         os.makedirs(uploads_dir)

#     # Construct the full path for the new file
#     save_path = os.path.join(uploads_dir, uploaded_file.name)
    
#     # Handle potential filename conflicts
#     if os.path.exists(save_path):
#         base, extension = os.path.splitext(uploaded_file.name)
#         counter = 1
#         while os.path.exists(save_path):
#             new_filename = f"{base}_{counter}{extension}"
#             save_path = os.path.join(uploads_dir, new_filename)
#             counter += 1

#     # Save the file
#     with open(save_path, 'wb+') as destination:
#         for chunk in uploaded_file.chunks():
#             destination.write(chunk)

#     return save_path


@login_required
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
    return render(request, 'feedback_form.html')
