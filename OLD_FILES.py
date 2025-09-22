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

# ------------------- CONFIG -------------------
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

# ------------------- CLIENTS -------------------
if not OPENAI_API_KEY or not ASSEMBLYAI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError('Missing required API keys. Please set OPENAI_API_KEY, ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY in environment.')

aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- HELPERS -------------------
def save_audio_file(tmp_path: str, target_folder: str = RECORDINGS_DIR) -> str:
    os.makedirs(target_folder, exist_ok=True)
    saved_path = os.path.join(target_folder, f"{int(time.time()*1000)}.wav")
    shutil.move(tmp_path, saved_path)  # works across drives
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
    # Fast path for WAV using wave module
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
    # The ElevenLabs SDK may return bytes-like or a generator; handle bytes
    result = retry_with_backoff(_call, step_name='TTS generate')
    # Normalize to bytes
    if hasattr(result, 'read'):
        mp3_bytes = result.read()
    elif isinstance(result, (bytes, bytearray)):
        mp3_bytes = bytes(result)
    else:
        # Fallback: attempt to bytes()
        mp3_bytes = bytes(result)
    if not mp3_bytes:
        raise RuntimeError('TTS returned empty audio. Please try again or switch voice.')
    # Convert to WAV
    import io
    wav_path = convert_mp3_bytes_to_wav_path(mp3_bytes, speed=speed)
    elapsed = time.perf_counter() - start
    logging.info(f"TTS generated in {elapsed*1000:.0f} ms -> {wav_path}")
    schedule_delete_file(wav_path, FILE_TTL_SECONDS)
    return wav_path

# ------------------- PIPELINE -------------------
def process_audio_pipeline(audio_filepath, *, voice: str = 'Rachel', speed: float = 1.0):
    recording_path = None
    try:
        if not audio_filepath:
            return 'ERROR: No audio provided.', None, None

        # Save audio across drives
        recording_path = save_audio_file(audio_filepath)

        # Validate
        ok, msg, duration_s = validate_audio_file(recording_path)
        if not ok:
            return f"ERROR: {msg}", None, None

        # 1️⃣ Transcribe with AssemblyAI v2 (retry)
        t_start = time.perf_counter()
        def _transcribe():
            return transcriber.transcribe(recording_path)
        transcript_resp = retry_with_backoff(_transcribe, step_name='Transcription')
        # Ensure completed
        if getattr(transcript_resp, 'status', 'completed') != 'completed':
            err_msg = getattr(transcript_resp, 'error', 'Transcription failed')
            return f'ERROR: {err_msg}', None, None
        transcript = (getattr(transcript_resp, 'text', '') or '').strip()
        if not transcript:
            return 'ERROR: Transcription returned empty text.', None, None
        t_elapsed = time.perf_counter() - t_start
        logging.info(f"Transcription ok in {t_elapsed*1000:.0f} ms (duration {duration_s:.2f}s)")

        # 2️⃣ GPT response (retry)
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
        # Post-process to enforce shortness
        if len(ai_text) > 1200:
            ai_text = ai_text[:1200].rsplit('.', 1)[0] + '.'
        g_elapsed = time.perf_counter() - g_start
        logging.info(f"LLM response ok in {g_elapsed*1000:.0f} ms, length={len(ai_text)}")

        # 3️⃣ TTS
        s_start = time.perf_counter()
        wav_path = tts_generate(ai_text, voice=voice, speed=speed)
        s_elapsed = time.perf_counter() - s_start
        logging.info(f"TTS ok in {s_elapsed*1000:.0f} ms -> {wav_path}")

        return transcript, wav_path, ai_text

    except Exception as e:
        logging.error(f"Pipeline exception: {e}")
        return f"ERROR: {str(e)}", None, None
    finally:
        # Clean up input recording promptly
        try:
            if recording_path and os.path.exists(recording_path):
                os.remove(recording_path)
                logging.info(f"Deleted input recording: {recording_path}")
        except Exception as exc:
            logging.warning(f"Failed to delete input recording {recording_path}: {exc}")

# ------------------- GRADIO UI -------------------
description_md = """
# AI Interview Voicebot
Speak your question and get a concise, interview-appropriate reply with TTS.

By using this demo you consent to send your audio to third-party services (AssemblyAI for transcription, OpenAI for text generation, ElevenLabs for TTS). Do not share sensitive information.
"""

with gr.Blocks(title='AI Interview Voicebot', theme=Soft(), css="""
:root{
  --radius-xxl:16px;
  --radius-lg:14px;
  --radius-md:12px;
  --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  /* Suggested soft palette */
  --bg-color: #000000;              /* Background (pure black) */
  --bg-secondary: #0F0F0F;          /* Slightly lighter panel on black */
  --text-primary: #E6E6E6;          /* Primary text */
  --text-secondary: #A8A8A8;        /* Secondary text */
  --primary-btn: #5EAEFF;           /* Primary buttons (accessible blue on black) */
  --secondary-btn: #6C757D;         /* Secondary buttons */
  --border-color: #2A2A2A;          /* Borders & dividers on black */
  --success-color: #2ECC71;         /* Success */
  --error-color: #E74C3C;           /* Error */
  --waveform-start: #5EAEFF;        /* Waveform gradient start */
  --waveform-end: #50E3C2;          /* Waveform gradient end */
}
.gradio-container{ background: var(--bg-color); color: var(--text-primary); font-family: var(--font-sans); font-size: 16.5px; line-height: 1.55; }
.gradio-container h1{ font-size: 1.8rem; letter-spacing: 0.2px; }
.gradio-container h2{ font-size: 1.3rem; letter-spacing: 0.2px; color: var(--text-secondary); }
.wrap{ max-width: 1040px; margin: 0 auto; }
.card{ border-radius: var(--radius-xxl); }
.status{ font-size: 0.95rem; color: var(--text-secondary); }

/* Inputs & boxes */
.gr-textbox, .gr-box, .gr-panel, .gr-accordion, .gradio-audio{ background: var(--bg-secondary); border-color: var(--border-color); border-radius: var(--radius-lg); }
.gr-input, .gr-textbox textarea{ color: var(--text-primary); border-radius: var(--radius-md); padding: 10px 12px; }

/* Buttons */
button{ border-radius: var(--radius-md) !important; font-weight: 600; }
button.primary, .gr-button-primary{ background: var(--primary-btn) !important; border-color: var(--primary-btn) !important; color: #0E1420 !important; }
.gr-button-secondary{ background: var(--secondary-btn) !important; border-color: var(--secondary-btn) !important; color: #fff !important; }
button:hover{ filter: brightness(1.05); }

/* Dividers & borders */
.gr-divider{ border-color: var(--border-color) !important; }

/* Increase recording section height */
#mic{ padding: 8px 0 6px; }
#mic .wrap, #mic .container{ min-height: 260px; }
#mic .audio-controls{ transform: scale(1.08); transform-origin: left center; }

/* Audio panel and pseudo-wave background gradient */
#mic .gradio-audio, .gradio-audio{ background: linear-gradient(90deg, color-mix(in srgb, var(--waveform-start), transparent 70%), color-mix(in srgb, var(--waveform-end), transparent 70%)); border-color: var(--border-color); border-radius: var(--radius-lg); }
""") as demo:
    gr.Markdown(description_md)

    with gr.Row(elem_classes=["wrap"]):
        # Left: Record + Transcript together
        with gr.Column(scale=1, min_width=460):
            mic_input = gr.Audio(label='🎙️ Record / Upload', type='filepath', sources=['microphone', 'upload'], max_length=MAX_RECORD_SECONDS, elem_id='mic', show_label=True, interactive=True)
            transcript_box = gr.Textbox(label='Transcript', lines=6, interactive=False, placeholder='Transcript will appear here…')
            with gr.Row():
                voice_dd = gr.Dropdown(label='Voice', choices=['Rachel','Adam','Bella','Antoni'], value='Rachel', interactive=True, info='Choose a voice')
                speed_slider = gr.Slider(minimum=0.8, maximum=1.2, step=0.05, value=1.0, label='TTS Speed', interactive=True)
            with gr.Row():
                submit_btn = gr.Button('💬 Generate Reply', variant='primary')
                clear_btn = gr.Button('♻️ Clear', variant='secondary')
                export_btn = gr.Button('⤓ Export Transcript', variant='secondary')
            status_box = gr.Markdown(value='Ready.', elem_classes=['status'])
        # Right: AI Voice + AI Response together
        with gr.Column(scale=1, min_width=460):
            ai_audio_out = gr.Audio(label='▶️ AI Voice', type='filepath')
            ai_text_box = gr.Textbox(label='AI Response', lines=8, interactive=False, placeholder='AI response text will appear here…')
            with gr.Row():
                thumb_up = gr.Button('👍 Helpful', variant='secondary')
                thumb_down = gr.Button('👎 Not helpful', variant='secondary')
        with gr.Accordion('Help', open=False):
            gr.Markdown('''- Record or upload a question up to 30s.\n- Click "Generate Reply" to get an interview-style answer.\n- Adjust voice and speed as needed.\n- Export transcript for notes.''')

    def handle_pipeline_stream(audio_path, voice, speed):
        yield gr.update(value='Validating audio...'), gr.update(), gr.update(), gr.update()
        transcript, wav_path, ai_text = process_audio_pipeline(audio_path, voice=voice, speed=speed)
        if transcript and not transcript.startswith('ERROR:'):
            yield gr.update(value='Transcribed. Generating reply...'), gr.update(), gr.update(), gr.update()
            yield gr.update(value='Reply generated. Creating voice...'), gr.update(), gr.update(), gr.update()
            yield gr.update(value='Done.'), wav_path, transcript, ai_text
        else:
            err = transcript or 'ERROR: Unknown'
            yield gr.update(value=err), None, None, None

    submit_btn.click(
        fn=handle_pipeline_stream,
        inputs=[mic_input, voice_dd, speed_slider],
        outputs=[status_box, ai_audio_out, transcript_box, ai_text_box]
    )

    def do_clear():
        return gr.update(value=None), gr.update(value=None), gr.update(value='Ready.'), gr.update(value=''), gr.update(value='')

    clear_btn.click(
        fn=do_clear,
        inputs=None,
        outputs=[mic_input, ai_audio_out, status_box, transcript_box, ai_text_box]
    )

    def do_export(transcript):
        if not transcript:
            return None
        path = os.path.join(EXPORTS_DIR, f"transcript_{int(time.time()*1000)}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        return path

    export_file = gr.File(label='Download Transcript', interactive=False)
    export_btn.click(
        fn=do_export,
        inputs=[transcript_box],
        outputs=[export_file]
    )

if __name__ == '__main__':
    demo.queue(max_size=2)
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)
