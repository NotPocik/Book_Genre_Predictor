import joblib
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import sklearn

# Отключение предупреждений
warnings.filterwarnings("ignore")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Функция для предобработки текста
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_genres(description):
    processed_text = preprocess_text(description)

    text_tfidf = vectorizer.transform([processed_text])
    return mlb.inverse_transform(model.predict(text_tfidf))
    


# Загрузка модели и вспомогательных объектов
model = joblib.load("genre_predictor_100.pkl")
vectorizer = joblib.load("tfidf_vectorizer_100.pkl")
mlb = joblib.load("mlb_100.pkl")


# Новый текст
new_description1 = '''

Jake Epping is a thirty-five-year-old high school English teacher in Lisbon Falls, Maine, who makes extra money teaching adults in the GED program. He receives an essay from one of the students—a gruesome, harrowing first person story about the night 50 years ago when Harry Dunning’s father came home and killed his mother, his sister, and his brother with a hammer. Harry escaped with a smashed leg, as evidenced by his crooked walk.

Not much later, Jake’s friend Al, who runs the local diner, divulges a secret: his storeroom is a portal to 1958. He enlists Jake on an insane—and insanely possible—mission to try to prevent the Kennedy assassination. So begins Jake’s new life as George Amberson and his new world of Elvis and JFK, of big American cars and sock hops, of a troubled loner named Lee Harvey Oswald and a beautiful high school librarian named Sadie Dunhill, who becomes the love of Jake’s life—a life that transgresses all the normal rules of time.
A tribute to a simpler era and a devastating exercise in escalating suspense, 11/22/63 is Stephen King at his epic best.
'''
new_description2 = '''
    
American researcher Jakub Mikanovskis, whose ancestors once lived in Eastern Europe, has done an in-depth study of the region's past. The book covers the history of this land from the 8th century to the present, touching on events related to the Christianization of these lands, the rise of the Polish-Lithuanian Commonwealth, the expansion and fall of the Ottoman, Austro-Hungarian and Russian Empires, the First and Second World Wars, the building of socialism and independence in the context of globalization.


'''



print("Предсказанные жанры 1:", predict_genres(new_description1))

print("Предсказанные жанры 2:", predict_genres(new_description2))
