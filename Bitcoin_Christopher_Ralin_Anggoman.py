"""Bitcoin_Streamlit_Christopher Ralin Anggoman.ipynb

# UAS Project Streamlit:
- **Nama:** Christopher Ralin Anggoman
- **Dataset:** [Dataset Harga Bitcoin] https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd
- **URL Website:** [Di isi jika web streamlit di upload]

## Menentukan Pertanyaan Bisnis

- Dapatkah model prediksi harga Bitcoin membantu para investor dalam pengambilan keputusan investasi jangka pendek atau jangka panjang?
- Sejauh mana tingkat keakuratan model dalam memprediksi perubahan harga Bitcoin dapat membantu dalam merancang strategi manajemen risiko?

## Import Semua Packages/Library yang Digunakan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

"""## Data Wrangling

### Gathering Data
"""

df = pd.read_csv('dataset/BTC-Daily.csv')
df.head()

df.shape
df.describe()

"""### Assessing Data"""

# Setelah memuat data, saya mengevaluasi kualitas dan integritas data untuk memahami apakah ada masalah atau kekurangan yang perlu diatasi. Melihat beberapa baris pertama dari data dan informasi statistik deskriptif adalah langkah awal dalam mengevaluasi data:
df.info()
df.head()

"""### Cleaning Data"""

# Setelah mengevaluasi data, saya melakukan pembersihan untuk memastikan keakuratannya. Langkah ini melibatkan penanganan nilai yang hilang, penyesuaian tipe data, dan langkah-langkah lainnya.

#### Penanganan Nilai yang Hilang
# Jika terdapat nilai yang hilang, kita dapat memutuskan untuk menghapus baris atau mengisi nilai tersebut dengan nilai yang sesuai.

# Menangani nilai yang hilang
df.dropna(inplace=True)

"""## Exploratory Data Analysis (EDA)

### Explore ...
"""

features = ['open', 'high', 'low', 'close']

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i])

# Remove overlapping axes
for ax in axes[len(features):]:
    ax.remove()

plt.show()

"""## Visualization & Explanatory Analysis

### Pertanyaan 1:

- Untuk menjawab pertanyaan ini, saya melakukan analisis visual terhadap fitur-fitur terkait harga Bitcoin, seperti 'open', 'high', 'low', dan 'close'. Visualisasi melibatkan distribusi dan box plot untuk memahami karakteristik dan variabilitas data

### Pertanyaan 2:

- Pertanyaan ini melibatkan analisis lebih lanjut terkait pertanyaan bisnis, dengan mengeksplorasi hubungan antara fitur-fitur yang telah diproses dan variabel target. Saya juga memeriksa apakah ada pola-pola tertentu yang dapat diamati melalui visualisasi untuk mendukung pengambilan keputusan

## Membuat Model
"""

# Split the date and time
splitted = df['date'].str.split(' ', expand=True)

# Extract date components
date_components = splitted[0].str.split('-', expand=True)

# Add date components to the DataFrame
df['year'] = date_components[0].astype('int')
df['month'] = date_components[1].astype('int')
df['day'] = date_components[2].astype('int')

# Extract time components
time_components = splitted[1].str.split(':', expand=True)

# Add time components to the DataFrame if needed
df['hour'] = time_components[0].astype('int')
df['minute'] = time_components[1].astype('int')
df['second'] = time_components[2].astype('int')

# Display the DataFrame
print(df)

df.head()
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()
df['open-close']  = df['open'] - df['close']
df['low-high']  = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

"""### Training Model"""

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

"""### Evaluasi Model"""

y_pred = models[0].predict(X_valid)
cm = confusion_matrix(Y_valid, y_pred)

# Plotting the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['class_0', 'class_1'],
            yticklabels=['class_0', 'class_1'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()

"""### Menyimpan Model"""

model = models[0]
joblib.dump(model, 'bitcoin_prediction.pkl')

"""## Conclusion

- Berdasarkan hasil analisis model prediksi harga Bitcoin, terlihat bahwa model dapat memberikan informasi yang berharga kepada para investor untuk mendukung pengambilan keputusan investasi. Namun, perlu diingat bahwa prediksi pasar keuangan selalu melibatkan risiko, dan keputusan investasi sebaiknya didasarkan pada analisis menyeluruh yang melibatkan lebih dari sekadar model prediksi.
- Model prediksi harga Bitcoin memberikan kontribusi yang positif dalam merancang strategi manajemen risiko. Tingkat keakuratan model dalam memprediksi perubahan harga dapat membantu para pelaku pasar untuk mengidentifikasi potensi risiko dan mengambil tindakan yang sesuai untuk melindungi portofolio investasi mereka. Namun, penting untuk diingat bahwa tidak ada model yang sempurna, dan faktor-faktor eksternal yang tidak dapat diprediksi juga dapat mempengaruhi pasar.
"""