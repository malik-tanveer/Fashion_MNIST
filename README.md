# Fashion-MNIST Image Classification

A small image-classification project that trains a convolutional neural network (CNN) to recognize clothing categories dataset. The project is implemented in a Jupyter notebook using TensorFlow/Keras, and includes a saved model and sample image assets.

## Contents

| File | Description |
| --- | --- |
| `Fashion_MNSIT_dataste.ipynb` | Notebook for loading and preprocessing Fashion-MNIST, training and evaluating the CNN, and trying image predictions. The filename is kept as it exists in the repository. |
| `mnist.keras` | Saved Keras model in the `.keras` format. |
| `images (2).jpg` | Image asset included with the project. |
| `jacket.png` | Image asset included with the project. |
| `phtot.jpg` | Image asset used by the notebook's sample prediction cell. |

## Requirements

- Python
- TensorFlow (including `tf.keras`)
- NumPy
- pandas
- Pillow
- Jupyter Notebook or JupyterLab

Install the packages in your Python environment:

```bash
python -m pip install tensorflow numpy pandas pillow jupyter
```

TensorFlow downloads Fashion-MNIST automatically the first time the notebook loads the dataset, so an internet connection is needed for that initial download.

## Run the notebook

1. Clone or download this repository.
2. Install the requirements above.
3. Start Jupyter from the repository directory:

   ```bash
   jupyter notebook
   ```

4. Open `Fashion_MNSIT_dataste.ipynb` and run the cells from top to bottom.

The notebook loads the dataset with `tensorflow.keras.datasets.fashion_mnist`. It contains 60,000 training images and 10,000 test images. Each image is a 28 × 28 grayscale image. The notebook reshapes images to `(28, 28, 1)` and scales pixel values to the range `[0, 1]`.

The sample prediction cell uses the Colab-specific path `/content/phtot.jpg`. When running locally, change that path to the location of an image file on your computer. The image predictor converts the image to grayscale, resizes it to 28 × 28, normalizes its pixel values, and predicts a class.

## Model

The notebook builds a sequential CNN with the following layers:

1. `Conv2D(8, kernel_size=3, activation="relu")`
2. `MaxPool2D(pool_size=2)`
3. `Flatten`
4. `Dense(64, activation="relu")`
5. `Dense(10, activation="softmax")`

It is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric. Training uses 5 epochs, batches of 32 images, and a 10% validation split from the training data.

The saved notebook output reports a test accuracy of approximately **89.51%** and a test loss of approximately **0.2908**. Results can vary when the model is trained again.

### Fashion-MNIST classes

The model's output indices map to these labels:

| Index | Category |
| ---: | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Predict with the saved model

After installing TensorFlow, NumPy, and Pillow, the saved model can be loaded to classify a local image:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model = tf.keras.models.load_model("mnist.keras")

image = Image.open("phtot.jpg").convert("L").resize((28, 28))
image_array = np.asarray(image, dtype=np.float32) / 255.0
image_array = image_array.reshape(1, 28, 28, 1)

probabilities = model.predict(image_array, verbose=0)
predicted_index = int(np.argmax(probabilities[0]))
print(class_names[predicted_index])
```

For best results, provide an image that resembles a Fashion-MNIST example: a single item of clothing, centered in a square image with a dark background and a light foreground. Ordinary product photos may differ substantially from the dataset's style and can be classified less reliably.

## Notes

- The notebook saves the trained model as `mnist.keras` in its current working directory. Running the final save cell again overwrites that file.
- The notebook imports pandas, although it is not required by the model's training or prediction logic.
- This repository does not currently include a separate Python application, command-line interface, or dependency lock file.
