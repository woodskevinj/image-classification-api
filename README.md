---
# 🖼️ Image Classification API

End-to-end deep learning project for **image classification** — built and deployed using **PyTorch** and **FastAPI**.  

This project demonstrates the **full applied ML lifecycle**:  
from data exploration and transfer learning to model serving and containerized deployment on AWS.

---

## 🧩 Project Overview

This project builds a **production-style image classification API** using a **pretrained CNN** (ResNet18).  
We fine-tune the model on the CIFAR-10 dataset, expose a `/predict` endpoint for inference, and deploy the system using containerized infrastructure.

---

## ⚙️ Tech Stack

| Component         | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| **PyTorch**       | Model training, fine-tuning, and inference         |
| **FastAPI**       | RESTful API for serving predictions                |
| **Docker**        | Containerization for reproducible deployment       |
| **Uvicorn**       | ASGI web server for FastAPI                        |
| **Jupyter**       | Interactive notebooks for training and exploration |
| **AWS ECS / ECR** | (Optional) Cloud hosting and deployment            |

---

## 📂 Project Structure

```
image-classification-api/
│
├── api/
│   ├── app.py                  # FastAPI app with /predict endpoint
│   └── utils/                  # (Optional) preprocessing helpers
│
├── data/
│   ├── .gitkeep                # Keeps folder tracked (data ignored)
│   ├── raw/                    # Original datasets (gitignored)
│   └── processed/              # Preprocessed data (gitignored)
│
├── models/
│   ├── .gitkeep                # Placeholder until model saved
│   └── best_model.pt           # Saved PyTorch weights (local only)
│
├── notebooks/
│   ├── 01_exploration.ipynb    # Dataset exploration + preprocessing
│   ├── 02_training.ipynb       # Model fine-tuning + evaluation
│   └── 05_explainability.ipynb # (Optional) SHAP or Grad-CAM analysis
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization for inference
├── .gitignore                  # Ignored files (data, models, cache)
├── README.md                   # Project documentation
└── ecs-task-def.json           # (Optional) AWS ECS task definition

```

---

## 🚀 Getting Started

```bash
# 1️⃣ Clone the repository
git clone https://github.com/woodskevinj/image-classification-api.git
cd image-classification-api

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the FastAPI app
uvicorn api.app:app --reload

# 4️⃣ Test the /predict endpoint
```

Send an image file for inference:

```bash
curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/predict
```

Expected JSON response:

```json
{
  "top1": {
    "label": "class_name",
    "score": 0.87
  }
}
```

---

## 🧠 Learning Focus

This project highlights core concepts required of an Applied ML Engineer:

- 🧩 Convolutional Neural Networks (CNNs) — for visual pattern extraction

- 🔁 Transfer Learning — leveraging pretrained ResNet architectures for small datasets

- ⚙️ Model Serving — deploying ML systems as APIs for real-time inference

- 🐳 Containerization — reproducible, portable environments for deployment

- ☁️ AWS Integration (Optional) — deploying via ECS with Docker and ECR

---

## 📊 Current Progress

| Phase                                | Description                                 | Status         |
| ------------------------------------ | ------------------------------------------- | -------------- |
| **Data Exploration & Preprocessing** | EDA, visualization, normalization constants | ✅ Completed   |
| **Model Training (ResNet18)**        | Fine-tuning pretrained CNN                  | ✅ Completed   |
| **API Development**                  | FastAPI `/predict` endpoint scaffold        | ✅ Completed   |
| **Containerization (Docker)**        | Dockerfile and ECS task definition setup    | 🕓 In progress |
| **Cloud Deployment (AWS ECS)**       | Push image to ECR and deploy                | 🕓 Upcoming    |

---

## 📅 Roadmap

- [x] Build dataset exploration and normalization notebook

- [x] Scaffold FastAPI app and /predict route

- [x] Train and save ResNet18 fine-tuned model

- [ ] Integrate trained model into API for inference

- [ ] Containerize with Docker

- [ ] Deploy to AWS ECS

---

👨‍💻 Author

- Kevin Woods
- Applied ML Engineer
- 🔗 GitHub: woodskevinj

🧾 License

- This project is open source under the MIT License.
