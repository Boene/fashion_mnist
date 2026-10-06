import tensorflow as tf
import matplotlib.pyplot as plt

from data import train_images, train_labels, test_images, test_labels
from sklearn.metrics import confusion_matrix

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_images,
    train_labels,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

labels = [
    "T-Shirt/Top",
    "Hose",
    "Pullover",
    "Kleid",
    "Mantel",
    "Sandale",
    "Hemd",
    "Sneaker",
    "Tasche",
    "Stiefelette"
]

### Ergebnisse und Analyse ###

predictions = model.predict(test_images, verbose=0)
predicted_labels = predictions.argmax(axis=1)

cm = confusion_matrix(test_labels, predicted_labels)

# Liste mit Indizes der Bilder, entsprechend der beiden Labels

indices = [
    i for i in range(len(test_labels))
    if test_labels[i] == 9 and predicted_labels[i] == 7
]

# Plot Accuracy und Loss
""" 
plt.plot(history.history["accuracy"])
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
 """
# Bilder mit realem vs vorhergesamten Label

"""
plt.figure(figsize=(10, 8))

for position, index in enumerate(indices[:12]):
    plt.subplot(3, 4, position + 1)

    plt.imshow(test_images[index], cmap="gray")

    plt.title(
        f"Index: {index}\n"
        f"echt: {labels[test_labels[index]]}\n"
        f"Netz: {labels[predicted_labels[index]]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show() 
"""

### Heatmap real vs vorhergesagt
""" 
print("Anzahl:", len(indices))
print("Indices:", indices)

fig, ax = plt.subplots(figsize=(10, 8))
image = ax.imshow(cm)

# Achsen definieren
ax.set_xlabel("Vorhergesagte Klasse")
ax.set_ylabel("Tatsächliche Klasse")
ax.set_xticks(range(10))
ax.set_yticks(range(10))

# Zahlen in die Felder schreiben
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j],
                ha="center",
                va="center")

# Legende rechts
legend_text = "\n".join(
    f"{i} – {label}"
    for i, label in enumerate(labels)
)

fig.text(
    0.92, 0.5,
    legend_text,
    va="center",
    fontsize=10
)

plt.tight_layout()
plt.show()
"""

###

print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)
