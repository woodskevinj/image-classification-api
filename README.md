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

image-classification-api/
├── data/
├── notebooks/
├── src/
├── api/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
