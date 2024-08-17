# Imports
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.callbacks import EarlyStopping

# Load and prepare the IMDB dataset
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, split=['train', 'test'])
train_data, test_data = dataset

# Convert dataset to pandas DataFrame
def dataset_to_dataframe(dataset):
    texts, labels = [], []
    for text, label in dataset:
        texts.append(text.numpy().decode('utf-8'))
        labels.append(label.numpy())
    return pd.DataFrame({'text': texts, 'label': labels})

train_df = dataset_to_dataframe(train_data)
test_df = dataset_to_dataframe(test_data)

# Exploratory Data Analysis (EDA)
print(train_df['label'].value_counts())

# Length of reviews
train_df['length'] = train_df['text'].apply(lambda x: len(x.split()))
plt.hist(train_df['length'], bins=50)
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Data Preprocessing
max_vocab_size = 10000
max_sequence_length = 500

tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(train_df['text'])
X_train = tokenizer.texts_to_sequences(train_df['text'])
X_test = tokenizer.texts_to_sequences(test_df['text'])

X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)
y_train = train_df['label'].values
y_test = test_df['label'].values

# Build the RNN with LSTM
model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, callbacks=[early_stopping], verbose=2)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
