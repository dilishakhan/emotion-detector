import pandas as pd # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Load dataset
data = pd.read_csv("dataset/fer2013.csv")

X = []
y = []

for i, row in data.iterrows():
    pixels = row['pixels']
    emotion = row['emotion']
    
    img = np.array(pixels.split(), dtype='float32')
    img = img.reshape(48, 48)
    
    X.append(img)
    y.append(emotion)

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Normalize
X = X / 255.0

# Reshape
X = X.reshape(-1, 48, 48, 1)

# One-hot encoding
y = to_categorical(y, num_classes=7)

# ✅ STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Save data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

np.save("X.npy", X)
np.save("y.npy", y)