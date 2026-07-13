# Study Buddy — Local Web-Based Learning Platform

A premium local-first learning platform and study companion designed specifically for system engineers, backend developers, and AI researchers. It runs completely locally on your machine, leveraging Ollama to drive explanations and solutions on your GPU.

## 📁 Directory Structure
The codebase is structured into clean, modular, and separate layers:
- **`backend/`**: Contains the FastAPI backend server implementation, SQLite database models, ChromaDB retriever, and the Ollama LLM orchestration logic.
- **`frontend/`**: Contains the web client assets:
  - `index.html`: Dashboard layout structure.
  - `style.css`: Premium dark modern glassmorphism aesthetic styling.
  - `app.js`: Dynamic client-side routing, progress tracking, and Markdown rendering.
- **`Books/`**: Directory where you drop your study materials in PDF format, categorized by subject folder (e.g. `Books/Git/`, `Books/OS/`).

---

## 🚀 Key Features

1. **Subjects Explainer & Topic-Wise Chat**:
   - **Checklist Learning**: Scan and index PDF textbooks inside your workspace library. The system extracts a full learning checklist of topics.
   - **Pedagogical Curriculum Sorting**: Topics are automatically sorted in logical sequence from beginner basics to advanced internals, using local AI.
   - **Direct Database Extraction**: Clicking a topic pulls context chunks directly from your indexed books and generates a deep, mechanics-focused explanation using Qwen 3.
   - **Topic-Wise Chat History**: Ask follow-up questions directly under any topic explanation. Chat history and previous context are saved separately per-topic.
   - **Memory & Recall**: Select any topic to instantly reload your previous discussion without re-generating the initial prompt, saving local GPU time.

2. **LeetCode Problem Solver & Teacher**:
   - Paste a LeetCode problem description.
   - The coding engine generates brute-force and optimized Python solutions.
   - **Beginner Code Walkthrough**: Under each solution, the explanation engine writes a thorough, word-by-word and line-by-line walk-through explaining the role of every Python keyword (`def`, `for`, `in`), operator, and syntax token. It avoids repetitive code listings by referencing line numbers.

3. **PDF Notes Export**:
   - Download beautiful, formatted PDF study notes of your discussions.
   - You can export the entire study session history, or filter down to a specific topic chat or LeetCode problem note.

---

## 💻 Hardware & Software Requirements

- **Operating System**: Windows / Linux / macOS
- **GPU Spec**: NVIDIA RTX 4060 (8GB VRAM) or equivalent (optimized for low-VRAM local execution)
- **Local AI Provider**: Ollama (installed and running)
- **Local Models**:
  - **Explainer Model**: `qwen3:8b` (default) or `deepseek-r1:8b`
  - **Coding Engine**: `qwen2.5-coder:7b`
  - **Embeddings Model**: `nomic-embed-text`

---

## 🛠️ Step-by-Step Setup

### 1. Download & Install Ollama
Ensure you have Ollama installed from [ollama.com](https://ollama.com). Open your terminal and pull the required models:
```bash
# Pull the explainer and coder models
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

### 2. Configure Python Environment
Install the Python dependencies in your terminal:
```bash
pip install -r requirements.txt
```

### 3. Load Your Books
Place the PDF textbooks you want to learn from into the `Books/` folder under their respective subject name subfolders:
```text
Books/
├── Git/
│   └── advanced_git.pdf
├── Python/
│   └── python_tricks.pdf
├── OS/
│   └── operating_systems.pdf
└── ...
```

---

## 🏃 How to Run the App

1. **Start the FastAPI Backend Server**:
   Navigate to the `backend/` directory and run:
   ```bash
   uvicorn main:app --port 8000
   ```

2. **Access the Web Dashboard**:
   Open your favorite web browser and go to:
   ```text
   http://localhost:8000
   ```
   *Note: The root path automatically redirects to `/static/index.html` where your dashboard is served.*

3. **Index Your Books**:
   - Go to the **Index & Settings** tab.
   - Click the **Index** button next to any subject.
   - The server will dynamically scan your PDF files, chunk them, calculate embeddings via Ollama, and store them in your local Chroma vector database.
   - You can monitor the real-time progress bar directly in the settings table.

4. **Learn & Chat**:
   - Switch to the **Subjects Explainer** tab, select a subject, and click a topic to study.
   - Type follow-up questions at the bottom of the viewport to drill down on mechanics.
   - Click the **Export PDF** button at any time to save your notes!
