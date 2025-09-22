# High-level pipeline sketch

1. Browser (Gradio UI)

   * User clicks **Record** → Gradio captures audio (WAV) locally.
   * User clicks **Stop** → Gradio uploads the recorded audio file to your Python backend (same process you already used).
   * UI shows immediate status updates (“Uploading…”, “Transcribing…”, “Generating…”, “Converting to speech…”, “Ready”).

2. Backend (Python / Gradio server)

   * Receive audio file from browser.
   * Validate input (size, duration).
   * Send audio file to **AssemblyAI Realtime or REST Transcription** endpoint (we’ll use REST final transcript per your “manual stop” choice).
   * Receive final transcript from AssemblyAI.
   * Call **OpenAI `gpt-3.5-turbo`** with a compact system prompt + user transcript to create the AI reply.
   * Call **ElevenLabs TTS** to generate an MP3 (or WAV) for the reply.
   * Convert ElevenLabs audio to WAV if needed, ensure normalization, save to temp file.
   * Return to Gradio: (transcript text, AI text, TTS file path or bytes).
   * UI plays audio via the Gradio audio component and displays text.

3. Browser plays the returned audio and displays transcript & AI text.
