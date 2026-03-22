from src.QDAClassifier import QDAClassifier
from src.NBClassifier import NBClassifier
from src.LDAClassifier import LDAClassifier
from  generate_dataset import generate_dataset


X, y = generate_dataset(10, 3)
classifier = QDAClassifier()
classifier.fit(X, y)
print(classifier.get_params())

X_test, _ = generate_dataset(2, 3)
print(classifier.predict_proba(X_test))
print(classifier.predict(X_test))