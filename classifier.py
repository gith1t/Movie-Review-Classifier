import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Завантаження ресурсів NLTK (користувач має зробити це один раз, як вказано в README)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("Завантажую ресурси NLTK (stopwords)...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Видалення HTML-тегів
    text = re.sub(r'<.*?>', '', text)
    # 2. Видалення всього, крім літер
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 3. Приведення до нижнього регістру
    text = text.lower()
    # 4. Токенізація
    tokens = word_tokenize(text)
    # 5. Видалення стоп-слів та слів довжиною < 3
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # 6. З'єднання слів у рядок
    return " ".join(filtered_tokens)

def main():

    # 2. Завантаження даних. Використовуємо відносний шлях, щоб скрипт працював у будь-кого.
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        print(f"Датасет успішно завантажено. Розмір: {df.shape}")
    except FileNotFoundError:
        print("Помилка: Файл 'IMDB Dataset.csv' не знайдено. Переконайтесь, що він у тій самій папці.")
        return

    # 3. Попередня обробка тексту
    print("Триває попередня обробка тексту... Це може зайняти кілька хвилин.")
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # 4. Векторизація тексту (TF-IDF)
    # Використовуємо ті ж параметри, що і в ноутбуці
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # 5. Розділення даних на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42, # Для відтворюваності результатів
        stratify=y       # Зберігаємо однакове співвідношення класів
    )

    # 6. Побудова та навчання моделі
    print("Навчання моделі логістичної регресії...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # 7. Оцінка моделі
    print("\n--- ОЦІНКА МОДЕЛІ ---")
    y_pred = model.predict(X_test)
    
    # Виводимо звіт, як у ноутбуці
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 8. Демонстрація на 5 випадкових прикладах
    print("\n--- ПРИКЛАДИ ПЕРЕДБАЧЕНЬ ---")
    # Беремо оригінальні тексти для наочності
    original_reviews = df.loc[y_test.index] 
    
    # Вибираємо 5 випадкових відгуків з тестової вибірки
    # random_state=50, як у вашому ноутбуці, для відтворюваності
    sample_indices = y_test.sample(5, random_state=50).index
    
    sample_texts = df['cleaned_review'][sample_indices]
    sample_vectors = vectorizer.transform(sample_texts)
    sample_predictions = model.predict(sample_vectors)

    for i, (index, pred_label_num) in enumerate(zip(sample_indices, sample_predictions)):
        original_row = original_reviews.loc[index]
        predicted_label = 'Positive' if pred_label_num == 1 else 'Negative'
        
        print(f"--- ВІДГУК #{i+1} ---")
        print(f"Текст: {original_row['review'][:300]}...")
        print(f"Справжній відгук: {original_row['sentiment']}")
        print(f"Передбачення моделі: {predicted_label}")
        print("-" * 20 + "\n")

# Точка входу до скрипту
if __name__ == "__main__":
    main()