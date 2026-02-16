# FloraVision AI Deployment Guide 🚀

Congratulations! Your FloraVision AI application is ready for production. This guide covers the most popular deployment options.

## 🔗 Prerequisites
1. **API Key**: Ensure you have a valid `GOOGLE_API_KEY` from Google AI Studio.
2. **Environment Variables**: Use a `.env` file or set the variables in your hosting provider's dashboard.

---

## ☁️ Option 1: Streamlit Community Cloud (Easiest)
1. **Push to GitHub**: Ensure your latest code is on a GitHub repository.
2. **Deploy**:
    - Go to [share.streamlit.io](https://share.streamlit.io/).
    - Connect your repo, branch: `main`, file: `app.py`.
3. **Secrets**: 
    - In Streamlit Cloud dashboard, go to **Settings > Secrets**.
    - Add your `GOOGLE_API_KEY`:
    ```toml
    GOOGLE_API_KEY = "your-api-key-here"
    ```

---

## 🐳 Option 2: Docker Deployment (Recommended for Servers)
Use Docker for consistent environments on AWS, Azure, Google Cloud, or a private VPS.

### Build the image:
```bash
docker build -t floravision-ai .
```

### Run the container:
```bash
docker run -p 8501:8501 --env-file .env floravision-ai
```
The app will be available at `http://localhost:8501`.

---

## 🏗️ Option 3: Manual Server Deployment (Linux/VPS)
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd FloraVision-AI
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run with nohup or systemd**:
   ```bash
   nohup streamlit run app.py &
   ```

---

## ⚡ Performance Tips
- **Caching**: The app uses `st.cache_data`. For server deployments, ensure the cache has enough disk/memory space.
- **Resources**: LangGraph and YOLO (simulated) are lightweight, but real YOLO detection requires more CPU/RAM.

Built with ❤️ by FloraVision AI
