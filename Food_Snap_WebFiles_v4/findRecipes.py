import requests

app_id = '4b360cb1'
app_key = 'f7a5b582e6b1eb6f71976deec62ffc84'

#Throw everything into a function
def find_recipes(*ingredients):
    recipes = []
    
    ## Query the API based on the number of ingredients ##
    if len(ingredients) >= 6:
        url = f'https://api.edamam.com/search?q={ingredients[0]},{ingredients[1]},{ingredients[2]},{ingredients[3]},{ingredients[4]},{ingredients[5]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()

    if len(ingredients) == 5:
        url = f'https://api.edamam.com/search?q={ingredients[0]},{ingredients[1]},{ingredients[2]},{ingredients[3]},{ingredients[4]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()

    if len(ingredients) == 4:
        url = f'https://api.edamam.com/search?q={ingredients[0]},{ingredients[1]},{ingredients[2]},{ingredients[3]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()

    if len(ingredients) == 3:
        url = f'https://api.edamam.com/search?q={ingredients[0]},{ingredients[1]},{ingredients[2]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()

    if len(ingredients) == 2:
        url = f'https://api.edamam.com/search?q={ingredients[0]},{ingredients[1]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()
        
    if len(ingredients) == 1:
        url = f'https://api.edamam.com/search?q={ingredients[0]}&app_id={app_id}&app_key={app_key}&from=0&to=3'
        response = requests.get(url)
        data = response.json()
    
    
    ## Grab the pertinent info ##
    for r in range(len(data['hits'])):
        recipe = data['hits'][r]['recipe']['label']
        url = data['hits'][r]['recipe']['url']
        image_url = data['hits'][r]['recipe']['image']
        calories = data['hits'][r]['recipe']['calories']
        recipe_info = {
            'recipe': recipe,
            'url': url,
            'image_url': image_url,
            'calories': calories
        }
        recipes.append(recipe_info)
        
        
    return recipes
