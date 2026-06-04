import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# 1. LOAD DATASET


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Resize for MobileNet
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

print("Training Images:", x_train.shape)
print("Testing Images :", x_test.shape)


# 2. CUSTOM HYBRID LOSS


class HybridLoss(tf.keras.losses.Loss):

    def __init__(
            self,
            alpha=1.0,
            gamma=2.0,
            lambda_penalty=0.01
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty

    def call(self, y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0
        )


        # FOCAL LOSS


        cross_entropy = -tf.reduce_sum(
            y_true * tf.math.log(y_pred),
            axis=1
        )

        pt = tf.reduce_sum(
            y_true * y_pred,
            axis=1
        )

        focal_loss = (
                self.alpha *
                tf.pow(1.0 - pt, self.gamma) *
                cross_entropy
        )


        # CONFIDENCE PENALTY


        entropy = -tf.reduce_sum(
            y_pred * tf.math.log(y_pred),
            axis=1
        )

        confidence_penalty = -entropy


        # HYBRID LOSS


        hybrid_loss = (
                focal_loss +
                self.lambda_penalty * confidence_penalty
        )

        return tf.reduce_mean(hybrid_loss)

# Create custom loss object

loss_function = HybridLoss(
    alpha=1.0,
    gamma=2.0,
    lambda_penalty=0.01
)


# 3. LOAD MOBILENET


base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# 4. BUILD MODEL


model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        10,
        activation='softmax'
    )
])


# 5. MODEL SUMMARY


model.summary()


# 6. COMPILE MODEL


model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=['accuracy']
)

# 7. TRAIN MODEL


history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# 8. ACCURACY GRAPH


plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('MobileNet CIFAR-10 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()


# 9. LOSS GRAPH


plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('MobileNet CIFAR-10 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("loss_graph.png")
plt.show()


# 10. PREDICTIONS


y_pred = model.predict(x_test)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)


# 11. CONFUSION MATRIX


cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig("confusion_matrix.png")
plt.show()

# 12. CLASSIFICATION REPORT

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=class_names
    )
)

# 13. PRECISION / RECALL / F1

precision = precision_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

print("\nPrecision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)

# 14. FINAL EVALUATION

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nFinal Test Accuracy :", accuracy)
print("Final Test Loss     :", loss)

# 15. SAVE MODEL

model.save(
    "mobilenet_cifar10_hybrid.keras"
)

print("\nModel saved successfully.")