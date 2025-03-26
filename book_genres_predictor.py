import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
from collections import Counter

# Загрузка ресурсов NLTK
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')
# Инициализация инструментов
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Функция для предобработки текста
def preprocess_text(text):
    # Удаляем пунктуацию и приводим к нижнему регистру
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов и лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)



# Загрузка данных из CSV
data = pd.read_csv("books_1.Best_Books_Ever.csv")

# Фильтрация книг на английском языке
data = data[data['language'] == 'English']

# Выбор определённых столбцов
data = data[['title', 'genres', 'description']]


# Удаление строк с пропущенными описаниями или жанрами
data = data.dropna(subset=['description', 'genres'])    

# Преобразование жанров из строкового формата в список
data['genres'] = data['genres'].apply(ast.literal_eval)



# Подсчёт частоты жанров
all_genres = [genre for sublist in data['genres'] for genre in sublist]
genre_counts = Counter(all_genres)

popular_genres = {genre: count for genre, count in genre_counts.items() if count >= 100}

sorted_genres = sorted(popular_genres.items(), key=lambda x: x[1], reverse=True)

for genre, count in sorted_genres:
    print(f"{genre}: {count}")

# Порог для редких жанров
threshold = 100

# Получение жанров, которые встречаются чаще порога
common_genres = {genre for genre, count in genre_counts.items() if count >= threshold}

# Удаление редких жанров из каждого списка
data['genres'] = data['genres'].apply(lambda genres: [g for g in genres if g in common_genres])


# Преобразование жанров в формат многомерного массива
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['genres'])
print(len(genres_encoded[0]))

# Применение предобработки к описаниям
data['description'] = data['description'].apply(preprocess_text)



# Признаки (описания) и метки (жанры)
x = data['description']
y = genres_encoded

# Разделение
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Векторизация текстов
vectorizer = TfidfVectorizer(max_features=10000)
x_train_tfidf = vectorizer.fit_transform(x_train)
#x_test_tfidf = vectorizer.transform(x_test)

# Обучение модели
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(x_train_tfidf, y_train)

# Оценка на тестовых данных
#y_pred = model.predict(x_test_tfidf)
#print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(model, "genre_predictor_100.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_100.pkl")
joblib.dump(mlb, "mlb_100.pkl")