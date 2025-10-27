# MNIST Quickstart (PyTorch)

A simple and clean machine learning project for handwritten digit recognition using a Convolutional Neural Network (CNN).  
Includes training, evaluation, confusion matrix visualization, and single-image prediction.

## 📦 Installation
```bash
git clone https://github.com/Btntpbs/mnist-quickstart.git
cd mnist-quickstart
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

🚀 Training
python train.py --epochs 3 --batch-size 128 --lr 0.001

Outputs:
models/mnist_cnn.pth
outputs/
  ├ acc_plot.png
  ├ confusion_matrix.png
  └ metrics.json

🔍 Predict
python predict.py --image sample.png
Predicted digit: 7

📁 Project Structure
mnist-quickstart
├ train.py
├ predict.py
├ requirements.txt
├ README.md
└ models/ / outputs/

✅ Features

CNN-based digit classifier using PyTorch

Auto MNIST download

Accuracy plots and confusion matrix

Configurable training via CLI args

Simple, reusable code for ML beginners

🧠 Future Improvements

Streamlit demo UI

Jupyter notebook tutorial

Stronger model architecture

📜 License

MIT License
