import os
import time
import logging
import shutil
import threading
from typing import Callable, Tuple, Optional, Any
import gradio as gr
from gradio.themes import Soft
from elevenlabs import generate
import assemblyai as aai
from pydub import AudioSegment, effects
from dotenv import load_dotenv
from openai import OpenAI
import io
load_dotenv()

# ------------------- CONFIG (keep your existing config) -------------------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

MAX_RECORD_SECONDS = 30
MAX_FILE_MB = 5
REQUEST_TIMEOUT_SECONDS = 30
RECORDINGS_DIR = 'intermediate_audio/recordings'
TTS_DIR = 'intermediate_audio/tts'
EXPORTS_DIR = 'intermediate_audio/exports'
TTS_SAMPLE_RATE = 16000
TTS_CHANNELS = 1
FILE_TTL_SECONDS = 300

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ------------------- LOGGING -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- CLIENTS (keep your existing clients) -------------------
if not OPENAI_API_KEY or not ASSEMBLYAI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError('Missing required API keys. Please set OPENAI_API_KEY, ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY in environment.')

aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- HELPERS (keep your existing helpers) -------------------
def save_audio_file(tmp_path: str, target_folder: str = RECORDINGS_DIR) -> str:
    os.makedirs(target_folder, exist_ok=True)
    saved_path = os.path.join(target_folder, f"{int(time.time()*1000)}.wav")
    shutil.move(tmp_path, saved_path)
    logging.info(f"Saved audio file: {saved_path}, size={os.path.getsize(saved_path)} bytes")
    return saved_path

def schedule_delete_file(path: str, delay_seconds: int = FILE_TTL_SECONDS) -> None:
    def _delete_later():
        try:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Deleted temp file after TTL: {path}")
        except Exception as exc:
            logging.warning(f"Failed to delete temp file {path}: {exc}")
    timer = threading.Timer(delay_seconds, _delete_later)
    timer.daemon = True
    timer.start()

def retry_with_backoff(func: Callable[[], Any], *, retries: int = 3, base_delay: float = 0.5, max_delay: float = 4.0, step_name: str = 'step'):
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                logging.error(f"{step_name} failed after {retries} retries: {exc}")
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logging.warning(f"{step_name} failed (attempt {attempt}/{retries}), retrying in {delay:.1f}s: {exc}")
            time.sleep(delay)

def validate_audio_file(path: str) -> Tuple[bool, Optional[str], Optional[float]]:
    if not os.path.exists(path):
        return False, 'Audio file not found.', None
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, f'File too large (> {MAX_FILE_MB}MB).', None
    duration_s: Optional[float] = None
    try:
        import wave
        if path.lower().endswith('.wav'):
            with wave.open(path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration_s = frames / float(rate) if rate else None
        if duration_s is None:
            audio = AudioSegment.from_file(path)
            duration_s = audio.duration_seconds
    except Exception:
        return False, 'Could not read audio file. Please record again.', None
    if duration_s is None:
        return False, 'Could not determine audio duration.', None
    if duration_s > MAX_RECORD_SECONDS + 0.2:
        return False, f'Recording too long (> {MAX_RECORD_SECONDS}s).', duration_s
    return True, None, duration_s

def convert_mp3_bytes_to_wav_path(mp3_bytes: bytes, *, speed: float = 1.0) -> str:
    os.makedirs(TTS_DIR, exist_ok=True)
    wav_path = os.path.join(TTS_DIR, f"{int(time.time()*1000)}.wav")
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
    audio = audio.set_frame_rate(TTS_SAMPLE_RATE).set_channels(TTS_CHANNELS)
    if speed and abs(speed - 1.0) > 1e-3:
        try:
            audio = effects.speedup(audio, playback_speed=speed)
        except Exception:
            pass
    audio.export(wav_path, format='wav')
    return wav_path

def tts_generate(text: str, *, voice: str = 'Rachel', speed: float = 1.0) -> str:
    start = time.perf_counter()
    def _call():
        return generate(
            api_key=ELEVENLABS_API_KEY,
            text=text,
            voice=voice,
            stream=False
        )
    result = retry_with_backoff(_call, step_name='TTS generate')
    if hasattr(result, 'read'):
        mp3_bytes = result.read()
    elif isinstance(result, (bytes, bytearray)):
        mp3_bytes = bytes(result)
    else:
        mp3_bytes = bytes(result)
    if not mp3_bytes:
        raise RuntimeError('TTS returned empty audio. Please try again or switch voice.')
    wav_path = convert_mp3_bytes_to_wav_path(mp3_bytes, speed=speed)
    elapsed = time.perf_counter() - start
    logging.info(f"TTS generated in {elapsed*1000:.0f} ms -> {wav_path}")
    schedule_delete_file(wav_path, FILE_TTL_SECONDS)
    return wav_path

def process_audio_pipeline(audio_filepath, *, voice: str = 'Rachel', speed: float = 1.0):
    recording_path = None
    try:
        if not audio_filepath:
            return 'ERROR: No audio provided.', None, None

        recording_path = save_audio_file(audio_filepath)
        ok, msg, duration_s = validate_audio_file(recording_path)
        if not ok:
            return f"ERROR: {msg}", None, None

        # Transcribe
        t_start = time.perf_counter()
        def _transcribe():
            return transcriber.transcribe(recording_path)
        transcript_resp = retry_with_backoff(_transcribe, step_name='Transcription')
        if getattr(transcript_resp, 'status', 'completed') != 'completed':
            err_msg = getattr(transcript_resp, 'error', 'Transcription failed')
            return f'ERROR: {err_msg}', None, None
        transcript = (getattr(transcript_resp, 'text', '') or '').strip()
        if not transcript:
            return 'ERROR: Transcription returned empty text.', None, None
        t_elapsed = time.perf_counter() - t_start
        logging.info(f"Transcription ok in {t_elapsed*1000:.0f} ms (duration {duration_s:.2f}s)")

        # GPT response
        g_start = time.perf_counter()
        def _chat():
            return openai_client.with_options(timeout=REQUEST_TIMEOUT_SECONDS).chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role':'system', 'content':'You are a confident job applicant answering interview questions. Keep responses concise (2-4 sentences).'},
                    {'role':'user', 'content': transcript}
                ],
                max_tokens=150,
                temperature=0.7
            )
        response = retry_with_backoff(_chat, step_name='LLM chat')
        ai_text = (response.choices[0].message.content or '').strip()
        if len(ai_text) > 1200:
            ai_text = ai_text[:1200].rsplit('.', 1)[0] + '.'
        g_elapsed = time.perf_counter() - g_start
        logging.info(f"LLM response ok in {g_elapsed*1000:.0f} ms, length={len(ai_text)}")

        # TTS
        s_start = time.perf_counter()
        wav_path = tts_generate(ai_text, voice=voice, speed=speed)
        s_elapsed = time.perf_counter() - s_start
        logging.info(f"TTS ok in {s_elapsed*1000:.0f} ms -> {wav_path}")

        return transcript, wav_path, ai_text

    except Exception as e:
        logging.error(f"Pipeline exception: {e}")
        return f"ERROR: {str(e)}", None, None
    finally:
        try:
            if recording_path and os.path.exists(recording_path):
                os.remove(recording_path)
                logging.info(f"Deleted input recording: {recording_path}")
        except Exception as exc:
            logging.warning(f"Failed to delete input recording {recording_path}: {exc}")

# ------------------- MINIMALIST GRADIO UI -------------------
custom_css = """
/* Clean, minimal styling */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 800px !important;
    margin: 0 auto !important;
    background: #f8fafc !important;
}

.main-container {
    background: white !important;
    border-radius: 20px !important;
    padding: 40px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1) !important;
    margin: 20px !important;
}

/* Hide Gradio branding */
footer { display: none !important; }
.gradio-container .gradio-container-4-10-0 { border: none !important; }

/* Main title */
.main-title {
    text-align: center !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #2d3748 !important;
    margin-bottom: 10px !important;
}

.main-subtitle {
    text-align: center !important;
    color: #718096 !important;
    font-size: 18px !important;
    margin-bottom: 40px !important;
}

/* Audio recorder styling */
.audio-container {
    margin: 30px 0 !important;
    padding: 30px !important;
    background: #f7fafc !important;
    border-radius: 16px !important;
    border: 2px dashed #e2e8f0 !important;
    text-align: center !important;
}

/* Primary button */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
}

/* Response area */
.response-area {
    background: #f7fafc !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin: 20px 0 !important;
    border: 1px solid #e2e8f0 !important;
}

/* Status text */
.status-text {
    text-align: center !important;
    color: #718096 !important;
    font-size: 16px !important;
    margin: 20px 0 !important;
    padding: 12px !important;
    background: #edf2f7 !important;
    border-radius: 8px !important;
}

/* Hide advanced controls initially */
.advanced-controls {
    margin-top: 20px !important;
}

/* Secondary buttons */
.secondary-btn {
    background: #edf2f7 !important;
    color: #4a5568 !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    margin: 0 5px !important;
}

/* Privacy notice */
.privacy-notice {
    font-size: 13px !important;
    color: #a0aec0 !important;
    text-align: center !important;
    margin-top: 30px !important;
    padding-top: 20px !important;
    border-top: 1px solid #e2e8f0 !important;
}

/* Clean up default gradio styling */
.gradio-audio {
    border-radius: 12px !important;
}

.gradio-textbox {
    border-radius: 12px !important;
}

/* Hide labels for cleaner look */
.audio-container label {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #4a5568 !important;
}
"""

with gr.Blocks(
    title="AI Interview Assistant",
    theme=Soft(),
    css=custom_css
) as demo:

    with gr.Column(elem_classes=["main-container"]):
        # Clean header
        gr.HTML('<h1 class="main-title">AI Interview Assistant</h1>')
        gr.HTML('<p class="main-subtitle">Speak your question and get interview-appropriate responses</p>')

        # Main recording interface
        with gr.Column(elem_classes=["audio-container"]):
            audio_input = gr.Audio(
                label="🎙️ Record your interview question (max 30 seconds)",
                type="filepath",
                sources=["microphone"],
                max_length=MAX_RECORD_SECONDS,
                show_label=True,
                interactive=True
            )

        # Single prominent button
        generate_btn = gr.Button(
            "Generate Interview Response",
            variant="primary",
            elem_classes=["generate-btn"],
            size="lg"
        )

        # Status feedback
        status_display = gr.HTML('<div class="status-text">Ready to help with your interview practice</div>')

        # Response area (hidden initially)
        with gr.Column(visible=False, elem_classes=["response-area"]) as response_section:
            gr.HTML('<h3 style="color: #4a5568; margin-bottom: 16px;">Your AI Response</h3>')

            # Audio response
            ai_audio = gr.Audio(
                label="🔊 Listen to your response",
                type="filepath",
                interactive=False
            )

            # Text response (collapsible)
            with gr.Accordion("📝 View Response Text", open=False):
                ai_text = gr.Textbox(
                    label="Response Text",
                    lines=4,
                    interactive=False,
                    show_copy_button=True
                )

                # Question transcript (for reference)
                question_transcript = gr.Textbox(
                    label="Your Question (Transcript)",
                    lines=2,
                    interactive=False,
                    show_copy_button=True
                )

        # Advanced controls (collapsible)
        with gr.Accordion("⚙️ Advanced Settings", open=False, elem_classes=["advanced-controls"]):
            with gr.Row():
                voice_choice = gr.Dropdown(
                    label="Voice",
                    choices=["Rachel", "Adam", "Bella", "Antoni"],
                    value="Rachel",
                    scale=1
                )
                speech_speed = gr.Slider(
                    label="Speech Speed",
                    minimum=0.8,
                    maximum=1.2,
                    value=1.0,
                    step=0.05,
                    scale=1
                )

        # Secondary actions
        with gr.Row():
            clear_btn = gr.Button("🔄 Start Over", variant="secondary", elem_classes=["secondary-btn"])

        # Privacy notice
        gr.HTML('''
        <div class="privacy-notice">
            This demo uses secure third-party services for processing.
            Please avoid sharing sensitive personal information.
        </div>
        ''')

    # ------------------- EVENT HANDLERS -------------------

    def handle_generate(audio_file, voice, speed):
        if not audio_file:
            return (
                '<div class="status-text" style="color: #e53e3e;">Please record a question first</div>',
                gr.update(visible=False),
                None, None, None
            )

        # Update status to processing
        yield (
            '<div class="status-text">🎧 Processing your audio...</div>',
            gr.update(visible=False),
            None, None, None
        )

        # Process audio
        transcript, wav_path, ai_response = process_audio_pipeline(
            audio_file, voice=voice, speed=speed
        )

        if transcript and not transcript.startswith('ERROR:'):
            # Success
            yield (
                '<div class="status-text" style="color: #38a169;">✅ Response generated successfully!</div>',
                gr.update(visible=True),
                wav_path,
                ai_response,
                transcript
            )
        else:
            # Error
            error_msg = transcript if transcript else "Unknown error occurred"
            yield (
                f'<div class="status-text" style="color: #e53e3e;">❌ {error_msg}</div>',
                gr.update(visible=False),
                None, None, None
            )

    def handle_clear():
        return (
            '<div class="status-text">Ready to help with your interview practice</div>',
            gr.update(visible=False),
            None, None, None, None
        )

    # Wire up events
    generate_btn.click(
        fn=handle_generate,
        inputs=[audio_input, voice_choice, speech_speed],
        outputs=[status_display, response_section, ai_audio, ai_text, question_transcript]
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[status_display, response_section, audio_input, ai_audio, ai_text, question_transcript]
    )

if __name__ == '__main__':
    demo.queue(max_size=2)
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        share=False,
        show_error=True
    )
