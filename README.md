from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import requests

url = 'https://raw.githubusercontent.com/abirami1998/NYU-Data-Science-Bootcamp-Spring-2024/main/Week%208/glass.csv'
response = requests.get(url)
with open('glass.csv', 'wb') as f:
    f.write(response.content)

# Load data
data = pd.read_csv('glass.csv')


X = data.drop(columns=['AI'])
y = data['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_proba = model.predict_proba(X_test_scaled)

al_proba = y_proba[:, 0]
default_threshold = 0.5
y_pred_default_al = (al_proba >= default_threshold).astype(int)


accuracy_default_al = accuracy_score(y_test, y_pred_default_al)
precision_default_al = precision_score(y_test, y_pred_default_al, average='weighted')
recall_default_al = recall_score(y_test, y_pred_default_al, average='weighted')

print("Default Threshold Metrics for 'Al' column:")
print(f"Accuracy: {accuracy_default_al:.4f}")
print(f"Precision: {precision_default_al:.4f}")
print(f"Recall: {recall_default_al:.4f}")

thresholds = [0.3, 0.4, 0.6, 0.7]
for threshold in thresholds:
    y_pred_custom_al = (al_proba >= threshold).astype(int)
    accuracy_custom_al = accuracy_score(y_test, y_pred_custom_al)
    precision_custom_al = precision_score(y_test, y_pred_custom_al, average='weighted')
    recall_custom_al = recall_score(y_test, y_pred_custom_al, average='weighted')
    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy_custom_al:.4f}, Precision: {precision_custom
