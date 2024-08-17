from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Load the dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Reshape and display an example image
example_index = 0
example_image = X.iloc[example_index].values.reshape(28, 28)
plt.imshow(example_image, cmap="gray")
plt.title(f"Label: {y[example_index]}")
plt.show()
