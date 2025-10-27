---
# 🖼️ Image Classification API

End-to-end deep learning project for image classification — built and deployed using
PyTorch and FastAPI.

This project demonstrates the full applied ML lifecycle from data exploration and
transfer learning to model serving and containerized deployment on AWS.
---

## 🧩 Project Overview

This API uses a fine-tuned **ResNet18** model (trained on CIFAR-10) to classify images into 10 object categories.  
It features endpoints for inference, logs inspection, model health, and metadata — designed to mirror a real-world applied ML deployment.

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
│ ├── app.py # Main production-ready API (with endpoints + model)
│ ├── utils/ # (Optional) helper modules
| └── local_app.py # Original minimal test/stub version for local dev
│
├── data/
│ ├── .gitkeep # Keeps data folder tracked (datasets ignored)
│ ├── raw/ # Local-only original datasets
│ └── processed/ # Local-only preprocessed data
│
├── models/
│ ├── .gitkeep
│ └── best_model.pt # Fine-tuned ResNet18 weights
│
├── notebooks/
│ ├── 01_exploration.ipynb # Dataset EDA + preprocessing
│ ├── 02_training.ipynb # Model training + evaluation
│ └── 05_explainability.ipynb # (Optional) SHAP / Grad-CAM analysis
│
├── logs/
│ └── predictions.log # API prediction logs
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
└── ecs-task-def.json # (Optional) AWS ECS task definition

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
  "top1_prediction": { "label": "cat", "confidence": 0.8723 },
  "top3_predictions": [
    { "label": "cat", "confidence": 0.8723 },
    { "label": "dog", "confidence": 0.0671 },
    { "label": "deer", "confidence": 0.0339 }
  ]
}
```

---

## 🧠 Learning Focus

This project highlights core concepts required of an Applied ML Engineer:

- 🧩 Convolutional Neural Networks (CNNs) — for visual pattern extraction

- 🔁 Transfer Learning — adapting pretrained ResNet18 to CIFAR-10

- 📈 Model Evaluation — training/validation loss tracking

- ⚙️ Model Serving — running inference via FastAPI

- 🐳 Containerization — reproducible, portable environments for deployment

- ☁️ AWS Integration (Optional) — deploying via ECS with Docker and ECR

---

## 🌐 API Endpoints

| Endpoint | Method | Description                                                              |
| -------- | ------ | ------------------------------------------------------------------------ |
| /predict | POST   | Upload an image for classification (returns top-1 and top-3 predictions) |
| /logs    | GET    | Retrieve recent prediction logs (?limit=10)                              |
| /health  | GET    | Quick system health and model readiness check                            |
| /info    | GET    | View model metadata (architecture, parameters, size, etc.)               |
| /        | GET    | Welcome message and API overview                                         |

---

## 🐳 Docker Usage

Build Docker Image

```bash
docker build -t image-classification-api .
```

Run Container

```bash
docker run -p 8000:8000 image-classification-api
```

Then open:

```arduino
http://127.0.0.1:8000/health
```

You should see:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "message": "API and model are ready for inference."
}
```

---

## 🧱 Docker Ignore Setup

.dockerignore ensures that unnecessary local files (data, logs, checkpoints, etc.) are excluded from Docker builds for a small and efficient image.

Example included in repo:

```bash
data/
logs/
notebooks/.ipynb_checkpoints/
venv/
.git/
```

---

## 📊 Current Progress

| Phase                                | Description                               | Status       |
| ------------------------------------ | ----------------------------------------- | ------------ |
| **Data Exploration & Preprocessing** | CIFAR-10 dataset setup and visualization  | ✅ Completed |
| **Model Training (ResNet18)**        | Fine-tuning pretrained CNNon CIFAR-10     | ✅ Completed |
| **API Development**                  | FastAPI app + model inference integration | ✅ Completed |
| **Logging & Health Monitoring**      | Logs, /health, /info endpoints added      | ✅ Completed |
| **Containerization (Docker)**        | Docker build + run configuration          | ✅ Completed |
| **Cloud Deployment (AWS ECS)**       | Push image to ECR and deploy              | 🔜 Next      |

---

## 📅 Roadmap

- [x] Complete EDA and preprocessing

- [x] Train and save fine-tuned ResNet18 model

- [x] Integrate inference into FastAPI /predict route

- [x] Add logging, /logs, /health, and /info endpoints

- [x] Containerize with Docker

- [ ] Deploy to AWS ECS

---

## ☁️ Deployment Status

- ✅ Dockerized successfully using a multi-stage build (~1 GB final image)
- ⚙️ AWS ECR upload attempted — image too large for current bandwidth limits
- 🧩 Next iteration: optimize dependency footprint (use `torch-cpu`, lighter base image)
- 🚀 ECS deployment workflow will follow in the next phase

---

## 💡 Developer Note

- The file `api/local_app.py` contains the original FastAPI scaffold (used for initial testing).
- The main production-ready API runs from `api/app.py`.

---

👨‍💻 Author

- Kevin Woods
- Applied ML Engineer
- 🔗 GitHub: woodskevinj

🧾 License

- This project is open source under the MIT License.
