from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserPreference(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preference')
    is_vegan = models.BooleanField(default=False)
    is_vegetarian = models.BooleanField(default=False)
    is_gluten_free = models.BooleanField(default=False)
    is_keto = models.BooleanField(default=False)
    preferred_cuisines = models.CharField(max_length=200, blank=True)  # Comma-separated values or JSON

    def __str__(self):
        return f'{self.user.username} Preferences'

class Ingredient(models.Model):
    name = models.CharField(max_length=100)
    expiration_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.name

class Recipe(models.Model):
    title = models.CharField(max_length=255)
    ingredients = models.ManyToManyField(Ingredient, related_name='recipes')
    cuisine = models.CharField(max_length=100, blank=True)
    diet = models.CharField(max_length=100, blank=True)
    preparation_time = models.IntegerField(help_text='Preparation time in minutes', null=True, blank=True)
    source_url = models.URLField(max_length=200, blank=True)

    def __str__(self):
        return self.title

class UserFeedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedback')
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name='feedback')
    rating = models.IntegerField(help_text='Rating from 1 to 5')
    comments = models.TextField(blank=True)

    def __str__(self):
        return f'{self.user.username} on {self.recipe.title}'

