# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Step 1: Load datasets
recipes = pd.read_csv('RAW_recipes_textProc_VecTag(th=5)_VecIng(th=500).csv')
interactions = pd.read_csv('RAW_interactions_textProc.csv')

# Step 2: Preprocess and link datasets
# Rename `id` column in recipes to match `recipe_id` in interactions
recipes = recipes.rename(columns={'id': 'recipe_id'})

# Merge recipes and interactions on `recipe_id`
data = interactions.merge(recipes, on='recipe_id', how='inner')

# Step 3: Encode user_id
# Convert user_id into a numeric feature
user_encoder = LabelEncoder()
data['user_id_encoded'] = user_encoder.fit_transform(data['user_id'])

# Step 4: Feature selection
# Select features including user_id_encoded, numerical recipe features, and tag/ingredient features
feature_columns = ['user_id_encoded', 'minutes', 'n_steps', 'n_ingredients'] + [
    col for col in recipes.columns if col.startswith('tag_') or col.startswith('ingredient_')
]

# Prepare feature matrix (X) and target variable (y)
X = data[feature_columns]
y = data['rating']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Print results
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")