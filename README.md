# 🖼️ Image Classification API

End-to-end deep learning project for building and serving an image classification model using **PyTorch** and **FastAPI**.

This project demonstrates transfer learning, model deployment, and serving predictions via a RESTful API.

---

## 🧩 Project Overview

The goal is to create a lightweight but production-style image classification API using a pretrained CNN (e.g., ResNet or MobileNet).

We'll fine-tune the model on a small dataset, expose a `/predict` endpoint that accepts images, and return the predicted class.

---

## ⚙️ Tech Stack

- **PyTorch** — Model training and transfer learning
- **FastAPI** — Lightweight RESTful inference API
- **Docker** — Containerization for easy deployment
- **Uvicorn** — ASGI server for FastAPI

---

## 📂 Folder Structure

```

image-classification-api/
├── data/
├── notebooks/
├── src/
├── api/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

1. **Clone the repo:**

   ```bash
   git clone https://github.com/your-username/image-classification-api.git
   cd image-classification-api

   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**

   ```bash
   uvicorn api.main:app --reload
   ```

4. **Test Prediction Endpoint:**
   Use `curl` or Postman to send a POST request to `http://127.0.0.1:8000/predict` with an image file.
   Or curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/predict

## 🧠 Learning Focus

This project emphasizes:

- Convolutional Neural Networks (CNNs) for visual pattern extraction

- Transfer Learning for faster convergence and reduced data needs

- Model Serving to deploy ML systems as APIs

## 📅 Next Steps

- [ ] Data Cleaning + EDA
- [ ] Implement model training with a pretrained CNN
- [ ] Add FastAPI prediction endpoint
- [ ] Docker Containerization
- [ ] AWS ECR Upload + Cleanup
- [ ] AWS ECS (Deployment)

Author: Kevin Woods, Applied ML Engineer

---
