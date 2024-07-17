import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

file_path = 'C:\\Users\\joshi\\Downloads\\Dataset .csv'
data = pd.read_csv(file_path)

print("Initial Data Preview:")
print(data.head())

print("\nMissing Values Before Handling:")
print(data.isnull().sum())

for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

print("\nMissing Values After Handling:")
print(data.isnull().sum())

data_encoded = pd.get_dummies(data, drop_first=True)

print("\nPreprocessed Data Preview:")
print(data_encoded.head())

def create_user_vector(preferences, data):
    user_vector = pd.DataFrame([preferences])
    user_vector_encoded = pd.get_dummies(user_vector, drop_first=True)
    user_vector_encoded = user_vector_encoded.reindex(columns=data.columns, fill_value=0)
    return user_vector_encoded

def get_recommendations(preferences, data):
    user_vector_encoded = create_user_vector(preferences, data)
    similarity_scores = cosine_similarity(data, user_vector_encoded)
    data['similarity'] = similarity_scores
    recommendations = data.sort_values(by='similarity', ascending=False)
    return recommendations.head(10)


sample_user_preferences = {
    'Cuisines': 'Mexican',  
    'Price range': '$$$',  
    'Aggregate rating': 4.5  
}

recommendations = get_recommendations(sample_user_preferences, data_encoded)
print("\nTop Recommendations:")
print(recommendations)

