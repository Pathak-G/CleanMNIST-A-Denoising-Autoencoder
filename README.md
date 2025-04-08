# CleanMNIST: A Denoising Autoencoder 🧠✨

This project demonstrates how a **Denoising Autoencoder** can effectively remove noise from MNIST digit images using deep learning.

## 🔍 Overview
We train an autoencoder neural network on the MNIST dataset to reconstruct clean images from noisy inputs. The model learns a compressed representation (encoding) and uses it to denoise the input data.

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## 🚀 How It Works
1. Load the MNIST dataset
2. Add Gaussian noise to the images
3. Train a deep autoencoder to denoise the images
4. Visualize original, noisy, and denoised outputs

## 📈 Training Performance
- Model trained over 50 epochs
- Binary cross-entropy loss used
- Successfully learned to reconstruct clean digits from noisy inputs

## 🖼️ Sample Output

_(You can include your own screenshot or use `plt.savefig()` to generate one automatically!)_

## 📁 Dataset
- [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

## 📎 License
MIT License

---

✨ Feel free to clone, try, and modify the project!
