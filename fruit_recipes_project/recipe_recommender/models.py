from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    age = models.IntegerField(null=True, blank=True)
    allergies = models.TextField(blank=True)
    dietary_restrictions = models.TextField(blank=True)
    favourite_cuisine = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.user.username}'s profile"


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
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user_feedbacks')
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name='feedback')
    rating = models.IntegerField(help_text='Rating from 1 to 5')
    comments = models.TextField(blank=True)


    def __str__(self):
        return f'{self.user.username} on {self.recipe.title}'

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    fruit_or_vegetable = models.CharField(max_length=100)
    feedback_type = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return f"{self.user.username} - {self.fruit_or_vegetable}"

