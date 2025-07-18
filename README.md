
# 😊 Facial Expression Recognition using CNN

This project builds a Convolutional Neural Network (CNN) to classify facial expressions like Happy, Sad, Angry, etc. from grayscale images using the FER-2013 dataset. It demonstrates the full pipeline of data loading, preprocessing, model building, training, and evaluation.

---

## 📁 Dataset

The dataset used is a structured version of **FER-2013**, split into:

```
/train
   ├── Angry
   ├── Disgust
   ├── Fear
   ├── Happy
   ├── Sad
   ├── Surprise
   └── Neutral

/test
   ├── Angry
   ├── Disgust
   ├── Fear
   ├── Happy
   ├── Sad
   ├── Surprise
   └── Neutral
```

Each subfolder contains 48x48 grayscale `.png` images labeled by emotion.

---

## 🧠 Model Architecture

- **Input Shape**: (48, 48, 1)
- **Convolutional Layers**:
  - Conv2D(128) → MaxPooling2D → Dropout(0.4)
  - Conv2D(256) → MaxPooling2D → Dropout(0.4)
  - Conv2D(512) → MaxPooling2D → Dropout(0.4)
  - Conv2D(512) → MaxPooling2D → Dropout(0.4)
- **Fully Connected Layers**:
  - Flatten
  - Dense(512) → Dropout(0.4)
  - Dense(256) → Dropout(0.3)
  - Output: Dense(7, activation='softmax') — for 7 emotion classes

---

## 🔧 Technologies Used

- Python
- NumPy, Pandas
- TensorFlow, Keras
- Matplotlib, Seaborn
- PIL (Python Imaging Library)
- Scikit-learn (Label Encoding)

---

## ⚙️ How to Run the Project (Kaggle Notebook)

1. Upload the `.ipynb` notebook to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Add the dataset in the same structure as shown above.
3. Attach dataset paths like:
   - `/kaggle/input/facial-expression-dataset/train/...`
   - `/kaggle/input/facial-expression-dataset/test/...`
4. Run the cells sequentially.

---

## 🧮 Training Details

- **Epochs**: 50
- **Batch Size**: 128
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy

### 🔍 Why Adam?
- Combines momentum + adaptive learning rate.
- Works well without manual tuning.
- Efficient and fast convergence.

### 📉 Why Categorical Crossentropy?
- Ideal for multi-class classification problems.
- Works with one-hot encoded labels and softmax outputs.
- Penalizes incorrect predictions more when the model is overconfident.

---

## 📊 Results & Evaluation

- Training and validation accuracy/loss plotted over 50 epochs.
- Model achieves good generalization across emotion classes.
- Predictions are visualized with ground truth comparison.

---

## 🧪 Predict on Custom Image

You can load any grayscale image and predict the emotion like this:

```python
from PIL import Image
import numpy as np

img = Image.open("your_image.png").convert('L').resize((48, 48))
img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
pred = model.predict(img_array)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Predicted Emotion:", emotion_labels[np.argmax(pred)])
```

---

## 🧠 Concept: Backpropagation

The CNN is trained using backpropagation — an algorithm that:
- Computes error between predicted and true label.
- Propagates this error backward through the network.
- Updates weights using the gradients to reduce future error.

---

## 💡 Concepts Covered

- Convolutional Neural Networks (CNNs)
- Image Preprocessing (resizing, grayscale, normalization)
- One-hot encoding
- Dropout regularization
- Backpropagation and weight optimization
- Visualization of model performance
- Inference on custom images

---

## 🚀 Future Improvements

- Add data augmentation for better generalization.
- Try pre-trained models like VGG16 or MobileNet.
- Add confusion matrix and precision-recall analysis.
- Build a simple web UI with Streamlit or Flask for real-time emotion detection.

---

## 🙋‍♀️ Author

**Prachi Karande**  
Computer Engineering, DY Patil  
Skills: Python | Deep Learning | CNN | TensorFlow | Keras | Image Processing  

---

## 📎 License

This project is licensed under the MIT License.  
Free to use for learning, academic, or research purposes.
