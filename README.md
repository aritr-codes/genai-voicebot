# VoiceBot GenAI

VoiceBot GenAI is a Python-based web application that provides a **voice-based interview assistant**. The app allows users to speak their questions, and it responds in a **natural, conversational voice** using AI models.

---

## Features

- **Voice Input & Output:** Users can ask questions via microphone and receive audio responses.
- **AI-Powered Conversation:** Uses modern NLP models to simulate interview-style Q&A.
- **Web Interface:** Built with **Gradio**, providing a simple and user-friendly UI.
- **Customizable Backend:** Easily extendable to use different AI models or TTS systems.

---

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Example packages:

- `gradio`
- `pydub`
- `elevenlabs` (for TTS)
- `assemblyai` (for ASR)
- `dotenv` (for environment variable management)

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/voicebot_genai.git
cd voicebot_genai
````

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy `.env.example` to `.env` and fill in API keys or other sensitive info. For example:

```
ASSEMBLYAI_API_KEY=your_api_key
ELEVENLABS_API_KEY=your_api_key
```

---

## Running the App

```bash
python app.py
```

* The app will launch on `http://0.0.0.0:<PORT>` (default port: 7860)
* On deployment platforms like **Render**, the port is automatically picked from the environment variable `PORT`.

---

## Deployment

* Can be deployed on **Render**, **Heroku**, or similar Python web hosting services.
* Ensure `.env` is properly configured on the deployment platform.
* Use:

```python
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
```

---

## Project Structure

```
voicebot_genai/
├── app.py          # Main Gradio app
├── backend.py      # Backend AI logic
├── requirements.txt
├── .env            # Environment variables (not tracked in Git)
└── README.md
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.
