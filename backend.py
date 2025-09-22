import os
import time
import logging
import shutil
import threading
import asyncio
from typing import Callable, Tuple, Optional, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import gradio as gr
from gradio.themes import Soft
from elevenlabs import generate, voices
import assemblyai as aai
from pydub import AudioSegment, effects
from dotenv import load_dotenv
from openai import OpenAI
import io
import hashlib
import json
from pathlib import Path
import psutil
import tempfile
import numpy as np
import soundfile as sf

load_dotenv()

# ------------------- CONFIG -------------------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

MAX_RECORD_SECONDS = 30
MAX_FILE_MB = 5
REQUEST_TIMEOUT_SECONDS = 45  # Increased for reliability

# Prefer persistent storage path on Spaces if available
BASE_PERSIST = os.getenv('PERSIST_DIR', '').strip()
if BASE_PERSIST:
    RECORDINGS_DIR = os.path.join(BASE_PERSIST, 'intermediate_audio/recordings')
    TTS_DIR = os.path.join(BASE_PERSIST, 'intermediate_audio/tts')
    EXPORTS_DIR = os.path.join(BASE_PERSIST, 'intermediate_audio/exports')
    CACHE_DIR = os.path.join(BASE_PERSIST, 'cache')
else:
    RECORDINGS_DIR = 'intermediate_audio/recordings'
    TTS_DIR = 'intermediate_audio/tts'
    EXPORTS_DIR = 'intermediate_audio/exports'
    CACHE_DIR = 'cache'
TTS_SAMPLE_RATE = 16000
TTS_CHANNELS = 1
FILE_TTL_SECONDS = 600  # Increased to 10 minutes
CACHE_TTL_HOURS = 24
MAX_CONCURRENT_REQUESTS = 3
MEMORY_CLEANUP_THRESHOLD_MB = 500

# Create directories
for dir_path in [RECORDINGS_DIR, TTS_DIR, EXPORTS_DIR, CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ------------------- ENHANCED LOGGING -------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('voicebot.log'),
            logging.StreamHandler()
        ]
    )
    # Reduce noise from third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()

# ------------------- PERFORMANCE MONITORING -------------------
@dataclass
class PerformanceMetrics:
    transcription_time: float = 0
    llm_time: float = 0
    tts_time: float = 0
    total_time: float = 0
    audio_duration: float = 0
    cache_hit: bool = False

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = 100  # Keep last 100 requests

    def add_metrics(self, metrics: PerformanceMetrics):
        self.metrics.append(metrics)
        if len(self.metrics) > self.max_metrics:
            self.metrics.pop(0)

    def get_avg_times(self) -> Dict[str, float]:
        if not self.metrics:
            return {}

        return {
            'avg_transcription': sum(m.transcription_time for m in self.metrics) / len(self.metrics),
            'avg_llm': sum(m.llm_time for m in self.metrics) / len(self.metrics),
            'avg_tts': sum(m.tts_time for m in self.metrics) / len(self.metrics),
            'avg_total': sum(m.total_time for m in self.metrics) / len(self.metrics),
            'cache_hit_rate': sum(1 for m in self.metrics if m.cache_hit) / len(self.metrics)
        }

perf_monitor = PerformanceMonitor()

# ------------------- CACHING SYSTEM -------------------
class ResponseCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cleanup_old_cache()

    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate cache key from text, voice, and speed"""
        content = f"{text.lower().strip()}_{voice}_{speed:.2f}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cleanup_old_cache(self):
        """Remove cache files older than CACHE_TTL_HOURS"""
        try:
            cutoff = datetime.now() - timedelta(hours=CACHE_TTL_HOURS)
            for file_path in self.cache_dir.glob("*.json"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    file_path.unlink(missing_ok=True)
                    # Also remove associated audio file
                    audio_path = file_path.with_suffix('.wav')
                    audio_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    def get(self, text: str, voice: str, speed: float) -> Optional[Tuple[str, str]]:
        """Get cached response and audio path"""
        cache_key = self._get_cache_key(text, voice, speed)
        cache_file = self.cache_dir / f"{cache_key}.json"
        audio_file = self.cache_dir / f"{cache_key}.wav"

        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check if cache is still valid
                cache_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=CACHE_TTL_HOURS):
                    logger.info(f"Cache hit for key: {cache_key[:8]}...")
                    # Return audio if it exists, otherwise empty string (text-only cache)
                    audio_path = data.get('audio_path') or str(audio_file)
                    if audio_path and Path(audio_path).exists():
                        return data.get('response', ''), audio_path
                    return data.get('response', ''), ''
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def set(self, text: str, voice: str, speed: float, response: str, audio_path: str):
        """Cache response and audio"""
        try:
            cache_key = self._get_cache_key(text, voice, speed)
            cache_file = self.cache_dir / f"{cache_key}.json"
            cached_audio = self.cache_dir / f"{cache_key}.wav"

            stored_audio_path = ""
            # Copy audio file to cache if provided and exists
            try:
                if audio_path and os.path.exists(audio_path):
                    shutil.copy2(audio_path, cached_audio)
                    stored_audio_path = str(cached_audio)
            except Exception as e:
                logger.warning(f"Cache audio copy skipped: {e}")

            # Save metadata
            data = {
                'text': text,
                'voice': voice,
                'speed': speed,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'audio_path': stored_audio_path
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Cached response for key: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

response_cache = ResponseCache()

# ------------------- MEMORY MANAGEMENT -------------------
def check_memory_usage():
    """Monitor memory usage and trigger cleanup if needed"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        if memory_mb > MEMORY_CLEANUP_THRESHOLD_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB, triggering cleanup")
            cleanup_temp_files()
            return True
        return False
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return False

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        cutoff_time = time.time() - FILE_TTL_SECONDS

        for directory in [RECORDINGS_DIR, TTS_DIR]:
            for file_path in Path(directory).glob("*.wav"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink(missing_ok=True)
                    logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

# ------------------- IMPROVED CLIENTS -------------------
if not OPENAI_API_KEY or not ASSEMBLYAI_API_KEY or not ELEVENLABS_API_KEY:
    raise RuntimeError('Missing required API keys. Please set OPENAI_API_KEY, ASSEMBLYAI_API_KEY, ELEVENLABS_API_KEY in environment.')

aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

# OpenAI client with better configuration
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=2
)

# ------------------- ENHANCED HELPERS -------------------
def save_audio_file(tmp_path: str, target_folder: str = RECORDINGS_DIR) -> str:
    """Save audio file with better error handling"""
    try:
        os.makedirs(target_folder, exist_ok=True)
        timestamp = int(time.time() * 1000)
        saved_path = os.path.join(target_folder, f"audio_{timestamp}.wav")

        # Use copy2 to preserve metadata, fallback to move
        try:
            shutil.copy2(tmp_path, saved_path)
            os.unlink(tmp_path)  # Clean up original
        except (OSError, shutil.SameFileError):
            shutil.move(tmp_path, saved_path)

        file_size = os.path.getsize(saved_path)
        logger.info(f"Saved audio: {saved_path} ({file_size} bytes)")
        return saved_path
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise

def schedule_delete_file(path: str, delay_seconds: int = FILE_TTL_SECONDS) -> None:
    """Schedule file deletion with better error handling"""
    def _delete_later():
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Deleted temp file: {path}")
        except Exception as exc:
            logger.warning(f"Failed to delete temp file {path}: {exc}")

    timer = threading.Timer(delay_seconds, _delete_later)
    timer.daemon = True
    timer.start()

def retry_with_backoff(
    func: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    step_name: str = 'operation'
):
    """Enhanced retry logic with exponential backoff"""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                logger.error(f"{step_name} failed after {retries} retries: {exc}")
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(f"{step_name} failed (attempt {attempt}/{retries}), retrying in {delay:.1f}s: {exc}")
            time.sleep(delay)

def validate_audio_file(path: str) -> Tuple[bool, Optional[str], Optional[float]]:
    """Enhanced audio validation with better error messages"""
    try:
        if not os.path.exists(path):
            return False, 'Audio file not found. Please try recording again.', None

        file_size = os.path.getsize(path)
        size_mb = file_size / (1024 * 1024)

        if size_mb > MAX_FILE_MB:
            return False, f'File too large ({size_mb:.1f}MB > {MAX_FILE_MB}MB). Please record a shorter message.', None

        if file_size < 1000:  # Less than 1KB
            return False, 'Audio file seems too small. Please ensure you recorded properly.', None

        # Get duration
        duration_s = None
        try:
            # Fast path for WAV files
            if path.lower().endswith('.wav'):
                import wave
                with wave.open(path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        duration_s = frames / float(rate)

            # Fallback to pydub for other formats
            if duration_s is None:
                audio = AudioSegment.from_file(path)
                duration_s = audio.duration_seconds

        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return False, 'Invalid audio format. Please try recording again.', None

        if duration_s is None or duration_s <= 0:
            return False, 'Could not process audio file. Please try recording again.', None

        if duration_s > MAX_RECORD_SECONDS + 1.0:  # Allow 1s buffer
            return False, f'Recording too long ({duration_s:.1f}s > {MAX_RECORD_SECONDS}s). Please record a shorter message.', duration_s

        if duration_s < 0.5:  # Too short
            return False, 'Recording too short. Please speak for at least 0.5 seconds.', duration_s

        logger.info(f"Audio validation passed: {duration_s:.2f}s, {size_mb:.2f}MB")
        return True, None, duration_s

    except Exception as e:
        logger.error(f"Audio validation error: {e}")
        return False, f'Error processing audio file: {str(e)}', None

def convert_mp3_bytes_to_wav_path(mp3_bytes: bytes, *, speed: float = 1.0) -> str:
    """Enhanced audio conversion with better error handling"""
    try:
        os.makedirs(TTS_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        wav_path = os.path.join(TTS_DIR, f"tts_{timestamp}.wav")

        # Load and process audio
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        audio = audio.set_frame_rate(TTS_SAMPLE_RATE).set_channels(TTS_CHANNELS)

        # Apply speed adjustment if needed
        if speed and abs(speed - 1.0) > 0.05:
            try:
                audio = effects.speedup(audio, playback_speed=speed)
                logger.info(f"Applied speed adjustment: {speed}x")
            except Exception as e:
                logger.warning(f"Speed adjustment failed, using original: {e}")

        # Normalize audio levels
        audio = effects.normalize(audio)

        # Export with optimized settings
        audio.export(wav_path, format='wav', parameters=["-ac", "1", "-ar", str(TTS_SAMPLE_RATE)])

        logger.info(f"Audio converted: {len(mp3_bytes)} bytes -> {wav_path}")
        return wav_path

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise RuntimeError(f"Failed to process TTS audio: {str(e)}")

def tts_generate(text: str, *, voice: str = 'Rachel', speed: float = 1.0) -> str:
    """Enhanced TTS with caching and better error handling"""
    start_time = time.perf_counter()

    try:
        # Check cache first
        cached = response_cache.get(f"tts_{text}", voice, speed)
        if cached:
            _, audio_path = cached
            if os.path.exists(audio_path):
                logger.info(f"TTS cache hit: {audio_path}")
                return audio_path

        # Resolve voice: accept either a name or a voice_id
        resolved_voice = voice
        try:
            vs = voices(api_key=ELEVENLABS_API_KEY)
            for v in vs:
                if getattr(v, 'voice_id', '') == voice or getattr(v, 'name', '') == voice:
                    resolved_voice = getattr(v, 'name', voice)
                    break
        except Exception as e:
            logger.warning(f"Could not resolve ElevenLabs voice, using provided value '{voice}': {e}")

        # Generate TTS
        def _call():
            # Keep arguments compatible with installed elevenlabs SDK version
            return generate(
                api_key=ELEVENLABS_API_KEY,
                text=text,
                voice=resolved_voice,
                stream=False,
            )

        result = retry_with_backoff(_call, step_name='TTS generation')

        # Handle different response types
        if hasattr(result, 'read'):
            mp3_bytes = result.read()
        elif isinstance(result, (bytes, bytearray)):
            mp3_bytes = bytes(result)
        else:
            mp3_bytes = bytes(result)

        if not mp3_bytes or len(mp3_bytes) < 100:
            raise RuntimeError('TTS returned empty or invalid audio')

        # Convert to WAV
        wav_path = convert_mp3_bytes_to_wav_path(mp3_bytes, speed=speed)

        # Cache the result (use resolved voice for key stability)
        response_cache.set(f"tts_{text}", resolved_voice, speed, text, wav_path)

        # Schedule cleanup
        schedule_delete_file(wav_path, FILE_TTL_SECONDS)

        elapsed = time.perf_counter() - start_time
        logger.info(f"TTS completed in {elapsed*1000:.0f}ms: {wav_path}")

        return wav_path

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"Voice generation failed: {str(e)}")

# ------------------- IMPROVED PIPELINE -------------------
def process_audio_pipeline(audio_filepath: str, *, voice: str = 'Rachel', speed: float = 1.0):
    """Enhanced processing pipeline with performance monitoring and caching"""
    recording_path = None
    metrics = PerformanceMetrics()
    pipeline_start = time.perf_counter()

    try:
        # Check memory usage
        check_memory_usage()

        if not audio_filepath:
            return 'ERROR: No audio provided.', None, None

        logger.info(f"Starting pipeline: voice={voice}, speed={speed}")

        # Save and validate audio
        recording_path = save_audio_file(audio_filepath)
        ok, msg, duration_s = validate_audio_file(recording_path)

        if not ok:
            return f"ERROR: {msg}", None, None

        metrics.audio_duration = duration_s or 0

        # Step 1: Transcription
        transcription_start = time.perf_counter()

        def _transcribe_primary():
            return transcriber.transcribe(
                recording_path,
                config=aai.TranscriptionConfig(
                    language_detection=True,
                    punctuate=True,
                    format_text=True
                )
            )

        try:
            transcript_resp = retry_with_backoff(_transcribe_primary, retries=2, step_name='Transcription')
        except Exception as e:
            transcript_resp = None

        needs_fallback = False
        if not transcript_resp or getattr(transcript_resp, 'status', 'error') != 'completed':
            # Typical Spaces error: language_detection cannot be performed on files with no spoken audio
            needs_fallback = True
        else:
            t_text = (getattr(transcript_resp, 'text', '') or '').strip()
            if not t_text:
                needs_fallback = True

        if needs_fallback:
            logger.info("Transcription fallback: forcing English without language detection")

            def _transcribe_fallback():
                return transcriber.transcribe(
                    recording_path,
                    config=aai.TranscriptionConfig(
                        language_detection=False,
                        language_code='en',
                        punctuate=True,
                        format_text=True
                    )
                )

            transcript_resp = retry_with_backoff(_transcribe_fallback, retries=2, step_name='Transcription (fallback)')

            if getattr(transcript_resp, 'status', 'completed') != 'completed':
                error_msg = getattr(transcript_resp, 'error', 'Transcription failed')
                return f'ERROR: Transcription failed - {error_msg}', None, None

            transcript = (getattr(transcript_resp, 'text', '') or '').strip()
            if not transcript:
                return 'ERROR: Could not understand the audio. Please speak more clearly.', None, None
        else:
            transcript = (getattr(transcript_resp, 'text', '') or '').strip()
            if not transcript:
                return 'ERROR: Could not understand the audio. Please speak more clearly.', None, None

        metrics.transcription_time = time.perf_counter() - transcription_start
        logger.info(f"Transcription completed in {metrics.transcription_time*1000:.0f}ms")

        # Check cache for LLM response
        cached_response = response_cache.get(transcript, "gpt", 1.0)
        if cached_response:
            ai_text, _ = cached_response
            metrics.cache_hit = True
            logger.info("LLM cache hit")
        else:
            # Step 2: LLM Generation
            llm_start = time.perf_counter()
            def _chat():
                return openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {
                            'role': 'system',
        'content': '''You are a person speaking in the first person.

Background:
You have over 2 years of experience as a Machine Learning and Data Engineer in startup environments, handling end-to-end ML lifecycle projects. You have deployed ML models for product categorization that boosted accuracy by around 20% and optimized database performance by approximately 25%. You excel in debugging and monitoring production systems, maintaining uptime above 99.9%. You are comfortable refactoring backend logic and collaborating across teams, with experience mentoring interns. You are hands-on with LLM tools like LangChain, LangGraph, and OpenRouter, and continually expanding your knowledge in LLMs and AI tooling.

Projects:
Your portfolio includes an ML Algorithm Visualizer (an interactive learning app), an AI-Generated Text Detector (NLP), a productionized Product Categorization Model and a Generative AI Voicebots.

Skills:
Proficient in Python, PyTorch, scikit-learn, Transformers, NLP, Pandas/Dask, SQL (MySQL/PostgreSQL), FastAPI/Flask, Git, Tableau, and Excel. Your core strengths lie in deployment, pipeline development, testing/debugging, and documentation.

Education:
Bachelor’s degree in Computer Science.

Speaking Style:
Respond in first person using concise, professional language, limited to 2–4 sentences. Your tone is confident, humble, and friendly, avoiding buzzwords and filler. Prefer responses highlighting concrete impacts with numbers and clear explanations. Use India-neutral global English.

Scope:
Answer interview-style questions about your background, strengths, growth areas, life story, misconceptions, and pushing your limits. Politely redirect if asked for sensitive or unknown information. For speculative queries, describe how you would find the answer. Never fabricate employers, dates, or credentials.

Formatting:
Use short paragraphs without bullet points. Limit answers to 2–4 sentences, with a maximum of 5 if absolutely necessary.
'''},
                        {'role': 'user', 'content': transcript}
                    ],
                    max_tokens=200,  # Increased slightly
                    temperature=0.7,
                    presence_penalty=0.1,  # Encourage diverse vocabulary
                    frequency_penalty=0.1   # Reduce repetition
                )

            response = retry_with_backoff(_chat, retries=2, step_name='LLM generation')
            ai_text = (response.choices[0].message.content or '').strip()

            # Post-process response
            if len(ai_text) > 1500:
                # Find natural break point
                sentences = ai_text.split('. ')
                truncated = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) > 1200:
                        break
                    truncated.append(sentence)
                    char_count += len(sentence) + 2
                ai_text = '. '.join(truncated) + '.'

            # Cache LLM response
            response_cache.set(transcript, "gpt", 1.0, ai_text, "")

            metrics.llm_time = time.perf_counter() - llm_start
            logger.info(f"LLM completed in {metrics.llm_time*1000:.0f}ms")

        # Step 3: TTS Generation
        tts_start = time.perf_counter()
        wav_path = tts_generate(ai_text, voice=voice, speed=speed)
        metrics.tts_time = time.perf_counter() - tts_start

        # Final metrics
        metrics.total_time = time.perf_counter() - pipeline_start
        perf_monitor.add_metrics(metrics)

        # Log performance summary
        logger.info(f"Pipeline completed in {metrics.total_time*1000:.0f}ms total")
        avg_times = perf_monitor.get_avg_times()
        if avg_times:
            logger.info(f"Avg times - Transcription: {avg_times.get('avg_transcription', 0)*1000:.0f}ms, "
                       f"LLM: {avg_times.get('avg_llm', 0)*1000:.0f}ms, "
                       f"TTS: {avg_times.get('avg_tts', 0)*1000:.0f}ms, "
                       f"Cache hit rate: {avg_times.get('cache_hit_rate', 0)*100:.1f}%")

        return transcript, wav_path, ai_text

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return f"ERROR: Processing failed - {str(e)}", None, None

    # finally:
    #     # Clean up input recording
    #     if recording_path and os.path.exists(recording_path):
    #         try:
    #             os.remove(recording_path)
    #             logger.debug(f"Cleaned up input: {recording_path}")
    #         except Exception as e:
    #             logger.warning(f"Failed to clean up input: {e}")

# Add periodic cleanup
def periodic_cleanup():
    """Run periodic maintenance tasks"""
    while True:
        try:
            time.sleep(300)  # Every 5 minutes
            cleanup_temp_files()
            check_memory_usage()
        except Exception as e:
            logger.warning(f"Periodic cleanup error: {e}")

# Start background cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

def save_numpy_audio(audio_tuple, target_folder: str = RECORDINGS_DIR) -> str:
    """Save numpy audio (sr, np.array) to wav file and return path"""
    try:
        os.makedirs(target_folder, exist_ok=True)
        sr, data = audio_tuple  # (sample_rate, np.array)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # convert to mono if stereo

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=target_folder)
        sf.write(tmp.name, data, sr)
        tmp.close()

        logger.info(f"Saved numpy audio -> {tmp.name}")
        return tmp.name
    except Exception as e:
        logger.error(f"Failed to save numpy audio: {e}")
        raise

logger.info("Enhanced AI Interview Voicebot backend initialized")

# Export the main function for the UI
__all__ = ['process_audio_pipeline', 'perf_monitor', 'save_numpy_audio']
