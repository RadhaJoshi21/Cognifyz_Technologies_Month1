import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

data = pd.read_csv('C:\\Users\\joshi\\Downloads\\Dataset .csv')

print("Dataset Loaded:")
print(data.head())

data.loc[:, 'Cuisines'] = data['Cuisines'].fillna('Unknown')

print("Missing values handled:")
print(data['Cuisines'].head())

label_encoders = {}
for column in ['Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color', 'Rating text']:
    le = LabelEncoder()
    data.loc[:, column] = le.fit_transform(data[column])
    label_encoders[column] = le

print("Categorical variables encoded:")
print(data.head())

data['Cuisines'] = data['Cuisines'].apply(lambda x: x.split(', '))
mlb = OneHotEncoder(sparse_output=False)
cuisines_encoded = mlb.fit_transform(data['Cuisines'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
cuisines_labels = mlb.categories_[0]

cuisines_df = pd.DataFrame(cuisines_encoded, columns=cuisines_labels)
data = pd.concat([data, cuisines_df], axis=1)

data.drop(['Restaurant ID', 'Cuisines'], axis=1, inplace=True)

print("Dataset after preprocessing:")
print(data.head())


X = data.drop(cuisines_labels, axis=1)  
y = data[cuisines_labels]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training and testing sets created:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

print("Model trained")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=cuisines_labels))
