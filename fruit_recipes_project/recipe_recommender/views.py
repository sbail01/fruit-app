from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import numpy as np  # Ensure NumPy is imported

# Load the Hugging Face model and feature extractor
model_name = "PedroSampaio/fruits-360-16-7"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

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
