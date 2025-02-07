# 🎭 Emotion Recognition using CNN & LSTM

## 📝 Overview
This project is an **AI-based Emotion Recognition System** that detects emotions from images using a **Deep Learning model (CNN + LSTM)**.  
The model classifies images into **7 emotions**:  
😡 Angry, 🤢 Disgust, 😨 Fear, 😃 Happy, 😐 Neutral, 😢 Sad, 😲 Surprise.

The system is trained on a dataset containing **28,821 training images** and **7,066 validation images** of human faces expressing various emotions.  
It utilizes **CNNs** for **spatial feature extraction** and **LSTMs** for **temporal pattern recognition**.

---

## 🚀 Features
- ✅ **Convolutional Neural Networks (CNN) for feature extraction**  
- ✅ **Long Short-Term Memory (LSTM) to analyze facial expressions over time**  
- ✅ **Supports real-time emotion detection via webcam**  
- ✅ **Pretrained weights available for direct testing**  
- ✅ **Data Augmentation & Batch Normalization applied**  
- ✅ **Trained on grayscale facial expression images (48x48 pixels)**  
- ✅ **Supports GPU acceleration for faster training**  

---

## 📂 Project Structure
```
Emotion-Recognition/
│── data/                        # Dataset directory
│   ├── train/                   # Training images
│   ├── validation/              # Validation images
│── models/                      # Trained model directory
│   ├── emotion_detection_model.h5  # Saved trained model
│── notebooks/                   # Jupyter Notebooks for experiments
│── scripts/                     # Python scripts for training & inference
│── requirements.txt              # Required dependencies
│── README.md                     # Project Documentation
│── app.py                        # Flask API for real-time emotion detection
│── train.py                      # Model training script
│── test.py                        # Model evaluation script
```

---

## 📊 Dataset
The model is trained on a **facial expression dataset** containing **7 emotion classes**:

| Emotion  | Training Images | Validation Images |
|----------|---------------|------------------|
| 😡 Angry   | 4,000         | 1,000            |
| 🤢 Disgust | 500           | 200              |
| 😨 Fear    | 4,500         | 1,200            |
| 😃 Happy   | 7,000         | 1,800            |
| 😐 Neutral | 6,000         | 1,500            |
| 😢 Sad     | 5,000         | 1,300            |
| 😲 Surprise| 1,800         | 600              |

### **Preprocessing:**
- ✔ Images resized to **48x48 pixels**  
- ✔ Converted to **grayscale** for consistency  
- ✔ Applied **data augmentation** (rotation, flipping, zooming)  
- ✔ Normalized pixel values to range **[0,1]**  

---

## 🏗 Model Architecture
The **hybrid CNN + LSTM model** consists of:

### **1️⃣ Convolutional Layers (Feature Extraction)**
- 4 **CNN layers** with `ReLU`, batch normalization, and dropout  
- Extracts spatial features from images  

### **2️⃣ LSTM Layer (Temporal Analysis)**
- Converts CNN features into sequences  
- Captures temporal dependencies  

### **3️⃣ Fully Connected Layers (Classification)**
- Dense layers with `Softmax` activation for classification  

### **Model Summary**
```
Layer (type)                Output Shape              Param #
=================================================================
Conv2D (64 filters, 3x3)     (None, 48, 48, 64)        640
BatchNormalization           (None, 48, 48, 64)        256
MaxPooling2D (2x2)           (None, 24, 24, 64)        0
Dropout (0.25)               (None, 24, 24, 64)        0
---------------------------------------------------------------
Conv2D (128 filters, 5x5)    (None, 24, 24, 128)       204928
BatchNormalization           (None, 24, 24, 128)       512
MaxPooling2D (2x2)           (None, 12, 12, 128)       0
Dropout (0.25)               (None, 12, 12, 128)       0
---------------------------------------------------------------
Conv2D (512 filters, 3x3)    (None, 12, 12, 512)       590336
BatchNormalization           (None, 12, 12, 512)       2048
MaxPooling2D (2x2)           (None, 6, 6, 512)         0
Dropout (0.25)               (None, 6, 6, 512)         0
---------------------------------------------------------------
Reshape                      (None, 9, 512)            0
SimpleRNN (64 units)         (None, 9, 64)             36928
LSTM (64 units)              (None, 64)                33024
---------------------------------------------------------------
Dense (256 units)            (None, 256)               16640
BatchNormalization           (None, 256)               1024
Dropout (0.25)               (None, 256)               0
---------------------------------------------------------------
Dense (512 units)            (None, 512)               131584
BatchNormalization           (None, 512)               2048
Dropout (0.25)               (None, 512)               0
---------------------------------------------------------------
Output Layer (7 classes)     (None, 7)                 3591
=================================================================
Total params: 3,385,415
Trainable params: 3,381,447
Non-trainable params: 3,968
```

---

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Muhammad-Zakriya-Tariq/AI-emotion-recognition.git
cd AI-emotion-recognition
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Train the Model**
```sh
python train.py
```

### **4️⃣ Test on Sample Images**
```sh
python test.py --image sample.jpg
```

### **5️⃣ Run Live Emotion Detection (Webcam)**
```sh
python app.py
```

---

## 🎯 Results
The model achieved **~36% accuracy** on validation data after **5 epochs**.  
Using **more epochs** and a **larger dataset** can further improve accuracy.

---

## ❓ FAQ
### 🔹 **Why did you use categorical cross-entropy loss?**
Since we are dealing with **multi-class classification (7 emotions)**, categorical cross-entropy is the best choice.

### 🔹 **How did you train your model?**
We trained our model using **Adam optimizer (learning rate = 0.0001)** with **batch size = 128**, early stopping, and learning rate reduction techniques.

### 🔹 **Why did you use CNN & LSTM together?**
CNN extracts **spatial features**, while LSTM captures **temporal dependencies**, making the model more robust.

### 🔹 **Can I use this model for real-time applications?**
Yes! Run `app.py` to detect emotions using your **webcam**.

---

## 📜 License
MIT License. Feel free to use and modify.

---

## 🤝 Contributing
Pull requests are welcome! If you find any issues, feel free to open an **Issue**.

---
# 🎭 Emotion Recognition using CNN & LSTM

## 📝 Overview
This project is an **AI-based Emotion Recognition System** that detects emotions from images using a **Deep Learning model (CNN + LSTM)**.  
The model classifies images into **7 emotions**:  
😡 Angry, 🤢 Disgust, 😨 Fear, 😃 Happy, 😐 Neutral, 😢 Sad, 😲 Surprise.

The system is trained on a dataset containing **28,821 training images** and **7,066 validation images** of human faces expressing various emotions.  
It utilizes **CNNs** for **spatial feature extraction** and **LSTMs** for **temporal pattern recognition**.

---

## 🚀 Features
- ✅ **Convolutional Neural Networks (CNN) for feature extraction**  
- ✅ **Long Short-Term Memory (LSTM) to analyze facial expressions over time**  
- ✅ **Supports real-time emotion detection via webcam**  
- ✅ **Pretrained weights available for direct testing**  
- ✅ **Data Augmentation & Batch Normalization applied**  
- ✅ **Trained on grayscale facial expression images (48x48 pixels)**  
- ✅ **Supports GPU acceleration for faster training**  

---

## 📂 Project Structure
```
Emotion-Recognition/
│── data/                        # Dataset directory
│   ├── train/                   # Training images
│   ├── validation/              # Validation images
│── models/                      # Trained model directory
│   ├── emotion_detection_model.h5  # Saved trained model
│── notebooks/                   # Jupyter Notebooks for experiments
│── scripts/                     # Python scripts for training & inference
│── requirements.txt              # Required dependencies
│── README.md                     # Project Documentation
│── app.py                        # Flask API for real-time emotion detection
│── train.py                      # Model training script
│── test.py                        # Model evaluation script
```

---

## 📊 Dataset
The model is trained on a **facial expression dataset** containing **7 emotion classes**:

| Emotion  | Training Images | Validation Images |
|----------|---------------|------------------|
| 😡 Angry   | 4,000         | 1,000            |
| 🤢 Disgust | 500           | 200              |
| 😨 Fear    | 4,500         | 1,200            |
| 😃 Happy   | 7,000         | 1,800            |
| 😐 Neutral | 6,000         | 1,500            |
| 😢 Sad     | 5,000         | 1,300            |
| 😲 Surprise| 1,800         | 600              |

### **Preprocessing:**
- ✔ Images resized to **48x48 pixels**  
- ✔ Converted to **grayscale** for consistency  
- ✔ Applied **data augmentation** (rotation, flipping, zooming)  
- ✔ Normalized pixel values to range **[0,1]**  

---

## 🏗 Model Architecture
The **hybrid CNN + LSTM model** consists of:

### **1️⃣ Convolutional Layers (Feature Extraction)**
- 4 **CNN layers** with `ReLU`, batch normalization, and dropout  
- Extracts spatial features from images  

### **2️⃣ LSTM Layer (Temporal Analysis)**
- Converts CNN features into sequences  
- Captures temporal dependencies  

### **3️⃣ Fully Connected Layers (Classification)**
- Dense layers with `Softmax` activation for classification  

### **Model Summary**
```
Layer (type)                Output Shape              Param #
=================================================================
Conv2D (64 filters, 3x3)     (None, 48, 48, 64)        640
BatchNormalization           (None, 48, 48, 64)        256
MaxPooling2D (2x2)           (None, 24, 24, 64)        0
Dropout (0.25)               (None, 24, 24, 64)        0
---------------------------------------------------------------
Conv2D (128 filters, 5x5)    (None, 24, 24, 128)       204928
BatchNormalization           (None, 24, 24, 128)       512
MaxPooling2D (2x2)           (None, 12, 12, 128)       0
Dropout (0.25)               (None, 12, 12, 128)       0
---------------------------------------------------------------
Conv2D (512 filters, 3x3)    (None, 12, 12, 512)       590336
BatchNormalization           (None, 12, 12, 512)       2048
MaxPooling2D (2x2)           (None, 6, 6, 512)         0
Dropout (0.25)               (None, 6, 6, 512)         0
---------------------------------------------------------------
Reshape                      (None, 9, 512)            0
SimpleRNN (64 units)         (None, 9, 64)             36928
LSTM (64 units)              (None, 64)                33024
---------------------------------------------------------------
Dense (256 units)            (None, 256)               16640
BatchNormalization           (None, 256)               1024
Dropout (0.25)               (None, 256)               0
---------------------------------------------------------------
Dense (512 units)            (None, 512)               131584
BatchNormalization           (None, 512)               2048
Dropout (0.25)               (None, 512)               0
---------------------------------------------------------------
Output Layer (7 classes)     (None, 7)                 3591
=================================================================
Total params: 3,385,415
Trainable params: 3,381,447
Non-trainable params: 3,968
```

---

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Muhammad-Zakriya-Tariq/AI-emotion-recognition.git
cd AI-emotion-recognition
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Train the Model**
```sh
python train.py
```

### **4️⃣ Test on Sample Images**
```sh
python test.py --image sample.jpg
```

### **5️⃣ Run Live Emotion Detection (Webcam)**
```sh
python app.py
```

---

## 🎯 Results
The model achieved **~36% accuracy** on validation data after **5 epochs**.  
Using **more epochs** and a **larger dataset** can further improve accuracy.

---

## ❓ FAQ
### 🔹 **Why did you use categorical cross-entropy loss?**
Since we are dealing with **multi-class classification (7 emotions)**, categorical cross-entropy is the best choice.

### 🔹 **How did you train your model?**
We trained our model using **Adam optimizer (learning rate = 0.0001)** with **batch size = 128**, early stopping, and learning rate reduction techniques.

### 🔹 **Why did you use CNN & LSTM together?**
CNN extracts **spatial features**, while LSTM captures **temporal dependencies**, making the model more robust.

### 🔹 **Can I use this model for real-time applications?**
Yes! Run `app.py` to detect emotions using your **webcam**.

---

## 📜 License
MIT License. Feel free to use and modify.

---

## 🤝 Contributing
Pull requests are welcome! If you find any issues, feel free to open an **Issue**.

---
