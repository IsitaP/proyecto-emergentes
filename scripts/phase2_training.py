from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    classification_report
)


# =========================
# RUTAS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "dataset_armero.tsv"
OUTPUT_DIR = BASE_DIR / "phase2_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CARGAR DATASET
# =========================

print("Cargando dataset...")
df = pd.read_csv(DATASET_PATH, sep="\t")

print("\nPrimeras filas:")
print(df.head())

print("\nColumnas del dataset:")
print(df.columns.tolist())

print("\nTamaño del dataset:")
print(df.shape)

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nConteo de muestras por clase:")
print(df["clase"].value_counts())


# =========================
# DEFINIR BANDAS Y CLASE
# =========================

features = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12"
]

target = "clase"

# Validar que las columnas existan
missing_columns = [col for col in features + [target] if col not in df.columns]

if missing_columns:
    raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")

X = df[features]
y = df[target]

classes = sorted(y.unique())

print("\nClases detectadas:")
print(classes)


# =========================
# DIVISIÓN TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)


# =========================
# MODELOS
# =========================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=42
        ))
    ]),

    "ANN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": GaussianNB()
}


# =========================
# ENTRENAMIENTO Y EVALUACIÓN
# =========================

results = []

for model_name, model in models.items():

    print("\n==============================")
    print(model_name)
    print("==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\nMatriz de confusión:")
    print(cm)

    print("\nOverall Accuracy:")
    print(accuracy)

    print("\nKappa:")
    print(kappa)

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Guardar matriz de confusión como imagen
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    matrix_filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    matrix_path = OUTPUT_DIR / matrix_filename

    plt.savefig(matrix_path, dpi=300)
    plt.close()

    results.append({
        "Model": model_name,
        "Overall Accuracy": accuracy,
        "Kappa": kappa,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


# =========================
# GUARDAR TABLA DE RESULTADOS
# =========================

results_df = pd.DataFrame(results)

print("\n==============================")
print("TABLA RESUMEN")
print("==============================")
print(results_df)

results_path = OUTPUT_DIR / "phase2_model_results.csv"
results_df.to_csv(results_path, index=False)

print("\nArchivos generados en:")
print(OUTPUT_DIR)