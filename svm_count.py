import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pre_process import preprocess_text

# Carregar os dados do CSV
data = pd.read_csv("DataSetForSCR_Final.csv", encoding="unicode_escape")

# !! SEM LIMPEZA
# Dividir os dados em features (X) e target (y)
X_sem_limp = data["Sentence"]
Y_sem_limp = data["Label"]

# Dividir os dados em conjunto de treinamento e teste
X_train_sem_limp, X_test_sem_limp, y_train_sem_limp, y_test_sem_limp = train_test_split(
    X_sem_limp, Y_sem_limp, test_size=0.2, random_state=42
)

# Vetorização dos dados de texto
vectorizer_sem_limp = CountVectorizer()
X_train_vectorized_sem_limp = vectorizer_sem_limp.fit_transform(X_train_sem_limp)
X_test_vectorized_sem_limp = vectorizer_sem_limp.transform(X_test_sem_limp)

# Treinar o modelo SVM
svm_sem_limp = SVC(kernel="linear")
svm_sem_limp.fit(X_train_vectorized_sem_limp, y_train_sem_limp)

# Avaliar o modelo
y_pred_sem_limp = svm_sem_limp.predict(X_test_vectorized_sem_limp)

# Métricas de avaliação
print("\n--- SEM LIMPEZA ---")
accuracy = accuracy_score(y_test_sem_limp, y_pred_sem_limp)
print(f"Acurácia: {accuracy:.2f}")

print("\nRelatório de classificação:")
print(classification_report(y_test_sem_limp, y_pred_sem_limp))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test_sem_limp, y_pred_sem_limp))

# !! C LIMPEZA PARCIAL
# Dividir os dados em features (X) e target (y)
X_c_limp_parc = data["Sentence"].str.lower()
X_c_limp_parc = X_c_limp_parc.str.replace("[^a-zA-Z0-9 ]", " ")
Y_c_limp_parc = data["Label"]

# Dividir os dados em conjunto de treinamento e teste
X_train_c_limp_parc, X_test_c_limp_parc, y_train_c_limp_parc, y_test_c_limp_parc = (
    train_test_split(X_c_limp_parc, Y_c_limp_parc, test_size=0.2, random_state=42)
)

# Vetorização dos dados de texto
vectorizer_c_limp_parc = CountVectorizer()
X_train_vectorized_c_limp_parc = vectorizer_c_limp_parc.fit_transform(
    X_train_c_limp_parc
)
X_test_vectorized_c_limp_parc = vectorizer_c_limp_parc.transform(X_test_c_limp_parc)

# Treinar o modelo Naive Bayes
svm_c_limp_parc = SVC(kernel="linear")
svm_c_limp_parc.fit(X_train_vectorized_c_limp_parc, y_train_c_limp_parc)

# Avaliar o modelo
y_pred_c_limp_parc = svm_c_limp_parc.predict(X_test_vectorized_c_limp_parc)

# Métricas de avaliação
print("\n--- LIMPEZA PARCIAL ---")
accuracy = accuracy_score(y_test_c_limp_parc, y_pred_c_limp_parc)
print(f"Acurácia: {accuracy:.2f}")

print("\nRelatório de classificação:")
print(classification_report(y_test_c_limp_parc, y_pred_c_limp_parc))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test_c_limp_parc, y_pred_c_limp_parc))


# !! C LIMPEZA
# Dividir os dados em features (X) e target (y)
X_c_limp = data["Sentence"].apply(preprocess_text)
Y_c_limp = data["Label"]

# Dividir os dados em conjunto de treinamento e teste
X_train_c_limp, X_test_c_limp, y_train_c_limp, y_test_c_limp = train_test_split(
    X_c_limp, Y_c_limp, test_size=0.2, random_state=42
)

# Vetorização dos dados de texto
vectorizer_c_limp = CountVectorizer()
X_train_vectorized_c_limp = vectorizer_c_limp.fit_transform(X_train_c_limp)
X_test_vectorized_c_limp = vectorizer_c_limp.transform(X_test_c_limp)

# Treinar o modelo Naive Bayes
svm_classifier_c_limp = SVC(kernel="linear")
svm_classifier_c_limp.fit(X_train_vectorized_c_limp, y_train_c_limp)

# Avaliar o modelo
y_pred_c_limp = svm_classifier_c_limp.predict(X_test_vectorized_c_limp)

# Métricas de avaliação
print("\n--- LIMPEZA ---")
accuracy = accuracy_score(y_test_c_limp, y_pred_c_limp)
print(f"Acurácia: {accuracy:.2f}")

print("\nRelatório de classificação:")
print(classification_report(y_test_c_limp, y_pred_c_limp))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test_c_limp, y_pred_c_limp))
