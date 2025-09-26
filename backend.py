import os
import time
import logging
import threading
import asyncio
from typing import Callable, Tuple, Optional, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import importlib
import hashlib
import json
import numpy as np
import soundfile as sf
import httpx
from typing import cast, TYPE_CHECKING, Any
from pathlib import Path

# Optional dependencies are imported lazily via importlib to avoid import-time failures
def _try_import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

_dotenv = _try_import('dotenv')
if _dotenv and hasattr(_dotenv, 'load_dotenv'):
    try:
        _dotenv.load_dotenv()
    except Exception:
        pass

# load .env if available
try:
    from dotenv import load_dotenv as _ld
    _ld()
except Exception:
    pass

# ------------------- CONFIG -------------------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

MAX_RECORD_SECONDS = 30
MAX_FILE_MB = 5
REQUEST_TIMEOUT_SECONDS = 45
TTS_SAMPLE_RATE = 22050  # Higher quality for better audio
TTS_CHANNELS = 1
CACHE_TTL_HOURS = 24
MAX_CONCURRENT_REQUESTS = 3
MEMORY_CLEANUP_THRESHOLD_MB = 300

# ------------------- LOGGING -------------------
def setup_logging():
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Attach one stream handler if none exist
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(logging.StreamHandler())
    # Reduce third-party noise
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('assemblyai').setLevel(logging.WARNING)
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
        self.max_metrics = 50  # Reduced for memory efficiency

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

# ------------------- IN-MEMORY CACHING -------------------
class InMemoryCache:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.max_entries = 100  # Limit memory usage
        self.access_times: Dict[str, float] = {}

    def _get_cache_key(self, text: str, voice: str = "", speed: float = 1.0) -> str:
        content = f"{text.lower().strip()}_{voice}_{speed:.2f}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cleanup_old_entries(self):
        """Remove oldest entries when cache is full"""
        if len(self.cache) <= self.max_entries:
            return

        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(self.cache) - self.max_entries + 10]]

        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def get(self, text: str, voice: str = "", speed: float = 1.0) -> Optional[Tuple[str, bytes]]:
        cache_key = self._get_cache_key(text, voice, speed)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            cache_time = datetime.fromisoformat(entry['timestamp'])

            if datetime.now() - cache_time < timedelta(hours=CACHE_TTL_HOURS):
                self.access_times[cache_key] = time.time()
                logger.info(f"Cache hit for key: {cache_key[:8]}...")
                return entry.get('response', ''), entry.get('audio_bytes', b'')
            else:
                # Remove expired entry
                self.cache.pop(cache_key, None)
                self.access_times.pop(cache_key, None)

        return None

    def set(self, text: str, voice: str, speed: float, response: str, audio_bytes: bytes = b''):
        cache_key = self._get_cache_key(text, voice, speed)

        self.cache[cache_key] = {
            'text': text,
            'voice': voice,
            'speed': speed,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'audio_bytes': audio_bytes
        }

        self.access_times[cache_key] = time.time()
        self._cleanup_old_entries()
        logger.info(f"Cached response for key: {cache_key[:8]}...")

memory_cache = InMemoryCache()

# ------------------- API CLIENTS (Lazy Init) -------------------
_transcriber: Any = None
_openai_client: Any = None
_cleanup_started: bool = False

def config_status() -> Dict[str, bool]:
    return {
        'OPENAI_API_KEY': bool(OPENAI_API_KEY),
        'ASSEMBLYAI_API_KEY': bool(ASSEMBLYAI_API_KEY),
        'ELEVENLABS_API_KEY': bool(ELEVENLABS_API_KEY),
    }

def is_configured() -> bool:
    st = config_status()
    return all(st.values())

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        aai = _try_import('assemblyai')
        if aai is None:
            raise RuntimeError('assemblyai package is not installed')
        if not ASSEMBLYAI_API_KEY:
            raise RuntimeError('ASSEMBLYAI_API_KEY is not set')
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        _transcriber = aai.Transcriber()
    return _transcriber

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        openai_mod = _try_import('openai')
        if openai_mod is None or not hasattr(openai_mod, 'OpenAI'):
            raise RuntimeError('openai package is not installed')
        if not OPENAI_API_KEY:
            raise RuntimeError('OPENAI_API_KEY is not set')
        _openai_client = openai_mod.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=2
        )
    return _openai_client

# ------------------- IN-MEMORY AUDIO UTILITIES -------------------
def numpy_to_wav_bytes(audio_tuple) -> bytes:
    """Convert (sr, np.array) numpy audio into WAV bytes (mono if needed) with chunked processing."""
    try:
        sr, data = audio_tuple

        # Early validation
        if data is None:
            raise RuntimeError("Audio data is None")

        # Ensure data is numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check for empty data
        if data.size == 0:
            raise RuntimeError("Audio data is empty")

        # Downsample if sample rate is unnecessarily high (reduces data size)
        if sr > 16000:
            # Downsample to 16kHz for speech (sufficient for transcription)
            from scipy import signal
            downsample_factor = sr // 16000
            if downsample_factor > 1:
                data = signal.decimate(data, downsample_factor, zero_phase=True)
                sr = sr // downsample_factor
                logger.info(f"Downsampled audio from {sr * downsample_factor}Hz to {sr}Hz")

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Convert to float32 if needed (more efficient than other dtypes)
        if data.dtype != np.float32:
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)

        # Auto-gain normalization for very quiet recordings
        peak = float(np.max(np.abs(data))) if data.size > 0 else 0.0
        if 0.001 < peak < 0.05:  # Only amplify if very quiet but not silent
            gain = min(0.25 / peak, 10.0)
            data = data * gain
            logger.info(f"Applied gain adjustment: {gain:.2f}x")

        # Ensure data is in valid range
        data = np.clip(data, -1.0, 1.0)

        # Trim silence from beginning and end to reduce file size
        # Find first and last non-silent samples (threshold = 0.01)
        threshold = 0.01
        non_silent = np.abs(data) > threshold
        if np.any(non_silent):
            indices = np.where(non_silent)[0]
            first_idx = max(0, indices[0] - int(0.1 * sr))  # Keep 100ms padding
            last_idx = min(len(data), indices[-1] + int(0.1 * sr))
            data = data[first_idx:last_idx]
            logger.info(f"Trimmed {(len(data) - (last_idx - first_idx)) / sr:.2f}s of silence")

        # Write to buffer with optimized settings
        buf = io.BytesIO()

        # Use lower quality for faster processing (still good for speech)
        sf.write(buf, data, sr, format='WAV', subtype='PCM_16')

        buf.seek(0)
        wav_bytes = buf.getvalue()

        logger.info(f"Converted numpy audio to WAV: {len(wav_bytes)} bytes, duration: {len(data)/sr:.2f}s")
        return wav_bytes

    except Exception as e:
        logger.error(f"Failed to convert numpy to wav bytes: {e}")
        raise RuntimeError(f"Audio conversion failed: {str(e)}")

def wav_bytes_to_numpy(wav_bytes: bytes):
    """Convert WAV bytes to (sr, np.int16 array) for Gradio numpy audio output."""
    try:
        buf = io.BytesIO(wav_bytes)
        buf.seek(0)
        data, sr = sf.read(buf, dtype='float32', always_2d=False)

        # Convert float32 [-1, 1] to int16 for Gradio
        data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)

        logger.info(f"Converted WAV bytes to numpy: sr={sr}, shape={data.shape}")
        return sr, data

    except Exception as e:
        logger.error(f"Failed to convert wav bytes to numpy: {e}")
        raise RuntimeError(f"Audio conversion failed: {str(e)}")

def validate_wav_bytes(wav_bytes: bytes) -> Tuple[bool, Optional[str], Optional[float]]:
    """Validate WAV bytes for size and duration without touching disk."""
    try:
        if not wav_bytes:
            return False, 'No audio data provided.', None

        size_mb = len(wav_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return False, f'Audio too large ({size_mb:.1f}MB > {MAX_FILE_MB}MB). Please record shorter audio.', None

        if len(wav_bytes) < 1000:
            return False, 'Audio seems too small. Please ensure you recorded properly.', None

        # Get duration and basic amplitude using soundfile
        try:
            buf = io.BytesIO(wav_bytes)
            with sf.SoundFile(buf) as f:
                num_frames = len(f)
                sr = f.samplerate
                duration_s = num_frames / sr
            buf.seek(0)
            data, _sr = sf.read(buf, dtype='float32', always_2d=False)
            # Compute RMS to detect silence
            if isinstance(data, np.ndarray):
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                rms = float(np.sqrt(np.mean(np.square(np.clip(data, -1.0, 1.0)))))
            else:
                rms = 0.0
        except Exception:
            return False, 'Invalid audio format. Please try recording again.', None

        if duration_s <= 0:
            return False, 'Could not process audio. Please try recording again.', None
        if duration_s > MAX_RECORD_SECONDS + 2.0:  # Allow buffer
            return False, f'Recording too long ({duration_s:.1f}s > {MAX_RECORD_SECONDS}s). Please record shorter audio.', duration_s
        if duration_s < 0.3:
            return False, 'Recording too short. Please speak for at least 0.3 seconds.', duration_s

        # Simple silence check (tolerate lower volumes on Spaces)
        if 'rms' in locals() and rms < 0.0005:
            return False, 'No speech detected in audio. Please speak louder or closer to the mic.', duration_s

        logger.info(f"Audio validation passed: {duration_s:.2f}s, {size_mb:.2f}MB")
        return True, None, duration_s

    except Exception as e:
        logger.error(f"Audio validation error: {e}")
        return False, f'Error processing audio: {str(e)}', None

# ------------------- API FUNCTIONS -------------------
def upload_to_assemblyai(wav_bytes: bytes) -> str:
    """Upload raw wav bytes to AssemblyAI and return upload_url."""
    try:
        if not ASSEMBLYAI_API_KEY:
            raise RuntimeError('ASSEMBLYAI_API_KEY is not configured')
        headers = {'authorization': ASSEMBLYAI_API_KEY}
        resp = httpx.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            content=wav_bytes,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()
        upload_url = data.get('upload_url', '')

        if not upload_url:
            raise RuntimeError("No upload URL received from AssemblyAI")

        logger.info(f"Uploaded {len(wav_bytes)} bytes to AssemblyAI")
        return upload_url

    except Exception as e:
        logger.error(f"AssemblyAI upload failed: {e}")
        raise RuntimeError(f"Failed to upload audio for transcription: {str(e)}")

def transcribe_audio_bytes(wav_bytes: bytes) -> str:
    """Transcribe audio from bytes using AssemblyAI."""
    try:
        upload_url = upload_to_assemblyai(wav_bytes)

        # Primary transcription attempt
        aai = _try_import('assemblyai')
        if aai is None:
            raise RuntimeError('assemblyai package is not installed')
        try:
            primary_config = aai.TranscriptionConfig(
                language_detection=True,
                punctuate=True,
                format_text=True
            )
            transcript = get_transcriber().transcribe(upload_url, config=primary_config)
        except Exception as e:
            logger.warning(f"Primary transcription failed with exception: {e}. Trying fallback without language detection.")
            fallback_config = aai.TranscriptionConfig(
                language_detection=False,
                language_code='en',
                punctuate=True,
                format_text=True
            )
            transcript = get_transcriber().transcribe(upload_url, config=fallback_config)

        # If primary returned non-completed status, retry with fallback once
        if transcript.status != aai.TranscriptStatus.completed:
            err_msg = getattr(transcript, 'error', '') or ''
            logger.warning(f"Primary transcription returned status {transcript.status}. Error: {err_msg}. Retrying with fallback config.")
            fallback_config = aai.TranscriptionConfig(
                language_detection=False,
                language_code='en',
                punctuate=True,
                format_text=True
            )
            transcript = get_transcriber().transcribe(upload_url, config=fallback_config)

        aai = _try_import('assemblyai')
        if aai is None:
            raise RuntimeError('assemblyai package is not installed')
        if transcript.status != aai.TranscriptStatus.completed:
            error_msg = getattr(transcript, 'error', 'Transcription failed')
            # Map "no spoken audio" to a clearer message
            if 'no spoken audio' in str(error_msg).lower() or 'no speech' in str(error_msg).lower():
                raise RuntimeError('No speech detected in audio')
            raise RuntimeError(f"Transcription failed: {error_msg}")

        text = (transcript.text or '').strip()
        if not text:
            raise RuntimeError("No speech detected in audio")

        logger.info(f"Transcription successful: {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")

def generate_llm_response(user_text: str) -> str:
    """Generate LLM response using OpenAI."""
    system_prompt = '''
    You are an early-stage Machine Learning and Data Engineer speaking in the first person.
    You have 2+ years of experience in startups, handling end-to-end ML lifecycle projects, deploying models that improved accuracy by ~20% and database performance by ~25%.
    You excel at debugging, monitoring production systems with >99.9% uptime, refactoring backend logic, and mentoring interns.
    You work with Python, PyTorch, scikit-learn, Transformers, SQL, FastAPI/Flask, and LLM tools like LangChain and LangGraph.
    Answer interview-style questions about your background, strengths, growth areas, and experiences.
    Respond concisely in 2–4 sentences, confident yet humble, focusing on concrete impacts.
    Use professional, India-neutral global English without filler or buzzwords.
    Politely redirect if asked for sensitive or unknown information, and avoid fabricating employers, dates, or credentials.
'''

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_text}
    ]

    try:
        # Attempt using SDK (supports most environments)
        client = get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=messages,  # type: ignore[arg-type]
            max_tokens=200,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        ai_text = (response.choices[0].message.content or '').strip()  # type: ignore[index]
        logger.info(f"LLM response generated: {len(ai_text)} characters")
        return ai_text
    except Exception as sdk_err:
        logger.warning(f"SDK chat.completions failed ({sdk_err}). Falling back to HTTP API...")
        if not OPENAI_API_KEY:
            raise RuntimeError('OPENAI_API_KEY is not set')
        url = 'https://api.openai.com/v1/chat/completions'
        payload = {
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            'messages': messages,
            'max_tokens': 200,
            'temperature': 0.7,
            'presence_penalty': 0.1,
            'frequency_penalty': 0.1,
        }
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        r = httpx.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        ai_text = (data['choices'][0]['message']['content'] or '').strip()
        logger.info(f"LLM response generated via HTTP: {len(ai_text)} characters")
        return ai_text

def generate_tts_audio_bytes(text: str, voice: str = 'Rachel', speed: float = 1.0) -> bytes:
    """Generate TTS and return WAV bytes entirely in memory."""
    try:
        # Generate TTS
        if not ELEVENLABS_API_KEY:
            raise RuntimeError('ELEVENLABS_API_KEY is not set')
        el_mod = _try_import('elevenlabs')
        if el_mod is None or not hasattr(el_mod, 'generate'):
            raise RuntimeError('elevenlabs package is not installed')
        result = el_mod.generate(api_key=ELEVENLABS_API_KEY, text=text, voice=voice, stream=False)

        # Handle different response types
        if hasattr(result, 'read'):
            mp3_bytes = result.read()
        elif isinstance(result, (bytes, bytearray)):
            mp3_bytes = bytes(result)
        else:
            mp3_bytes = bytes(result)

        if not mp3_bytes or len(mp3_bytes) < 100:
            raise RuntimeError('TTS returned empty or invalid audio')

        # Convert MP3 to WAV using pydub in memory
        pd = _try_import('pydub')
        if pd is None or not hasattr(pd, 'AudioSegment'):
            raise RuntimeError('pydub package is not installed')
        mp3_buffer = io.BytesIO(mp3_bytes)
        audio_segment = pd.AudioSegment.from_file(mp3_buffer, format='mp3')

        # Optimize audio settings
        audio_segment = audio_segment.set_frame_rate(TTS_SAMPLE_RATE).set_channels(TTS_CHANNELS)

        # Apply speed adjustment if needed
        if speed and abs(speed - 1.0) > 0.05:
            try:
                effects_mod = _try_import('pydub.effects')
                if effects_mod is None or not hasattr(effects_mod, 'speedup'):
                    raise RuntimeError('pydub.effects not available')
                audio_segment = effects_mod.speedup(audio_segment, playback_speed=speed)
                logger.info(f"Applied speed adjustment: {speed}x")
            except Exception as e:
                logger.warning(f"Speed adjustment failed: {e}")

        # Normalize audio
        effects_mod = _try_import('pydub.effects')
        if effects_mod is not None and hasattr(effects_mod, 'normalize'):
            audio_segment = effects_mod.normalize(audio_segment)

        # Export to WAV bytes
        wav_buffer = io.BytesIO()
        audio_segment.export(
            wav_buffer,
            format='wav',
            parameters=["-ac", "1", "-ar", str(TTS_SAMPLE_RATE)]
        )
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.getvalue()

        logger.info(f"TTS generated: {len(wav_bytes)} bytes")
        return wav_bytes

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise RuntimeError(f"Voice generation failed: {str(e)}")

# ------------------- MAIN PROCESSING PIPELINE -------------------
def process_audio_pipeline_memory(wav_bytes: bytes, voice: str = '3gsg3cxXyFLcGIfNbM6C', speed: float = 1.0):
    """
    Complete in-memory processing pipeline for Spaces.
    Returns: (transcript, (sample_rate, numpy_array), ai_response)
    """
    metrics = PerformanceMetrics()
    pipeline_start = time.perf_counter()

    try:
        logger.info(f"Starting in-memory pipeline: voice={voice}, speed={speed}")

        if not wav_bytes:
            return 'ERROR: No audio provided.', None, None

        # Validate audio
        ok, msg, duration_s = validate_wav_bytes(wav_bytes)
        if not ok:
            return f"ERROR: {msg}", None, None

        metrics.audio_duration = duration_s or 0

        # Step 1: Transcription
        transcription_start = time.perf_counter()
        try:
            transcript = transcribe_audio_bytes(wav_bytes)
        except Exception as e:
            return f"ERROR: {str(e)}", None, None

        metrics.transcription_time = time.perf_counter() - transcription_start
        logger.info(f"Transcription completed in {metrics.transcription_time*1000:.0f}ms")

        # Step 2: Check cache for LLM response
        cached_response = memory_cache.get(transcript, "gpt", 1.0)
        if cached_response:
            ai_text, _ = cached_response
            metrics.cache_hit = True
            logger.info("LLM cache hit")
        else:
            # Generate LLM response
            llm_start = time.perf_counter()
            try:
                ai_text = generate_llm_response(transcript)
                memory_cache.set(transcript, "gpt", 1.0, ai_text, b'')
            except Exception as e:
                return f"ERROR: {str(e)}", None, None

            metrics.llm_time = time.perf_counter() - llm_start
            logger.info(f"LLM completed in {metrics.llm_time*1000:.0f}ms")

        # Step 3: Check cache for TTS
        cached_tts = memory_cache.get(f"tts_{ai_text}", voice, speed)
        if cached_tts:
            _, tts_wav_bytes = cached_tts
            metrics.cache_hit = True
            logger.info("TTS cache hit")
        else:
            # Generate TTS
            tts_start = time.perf_counter()
            try:
                tts_wav_bytes = generate_tts_audio_bytes(ai_text, voice=voice, speed=speed)
                memory_cache.set(f"tts_{ai_text}", voice, speed, ai_text, tts_wav_bytes)
            except Exception as e:
                return f"ERROR: {str(e)}", None, None

            metrics.tts_time = time.perf_counter() - tts_start
            logger.info(f"TTS completed in {metrics.tts_time*1000:.0f}ms")

        # Convert TTS bytes to numpy format for Gradio
        try:
            sr, numpy_audio = wav_bytes_to_numpy(tts_wav_bytes)
        except Exception as e:
            return f"ERROR: Audio format conversion failed: {str(e)}", None, None

        # Final metrics
        metrics.total_time = time.perf_counter() - pipeline_start
        perf_monitor.add_metrics(metrics)

        logger.info(f"Pipeline completed successfully in {metrics.total_time*1000:.0f}ms")
        return transcript, (sr, numpy_audio), ai_text

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return f"ERROR: Processing failed - {str(e)}", None, None

# ------------------- LEGACY FUNCTIONS FOR COMPATIBILITY -------------------
def process_audio_pipeline(*args, **kwargs):
    """Legacy function - redirects to memory-based pipeline"""
    logger.warning("process_audio_pipeline called - this function is deprecated for Spaces")
    return "ERROR: File-based processing not supported in Spaces environment", None, None

def save_numpy_audio(*args, **kwargs):
    """Legacy function - not needed for in-memory processing"""
    logger.warning("save_numpy_audio called - not needed for in-memory processing")
    return ""

# ------------------- CLEANUP -------------------
def cleanup_memory():
    """Clean up memory periodically"""
    try:
        # Clear old cache entries
        memory_cache._cleanup_old_entries()
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def _start_cleanup_thread_once():
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True
    def _runner():
        while True:
            time.sleep(300)
            cleanup_memory()
    t = threading.Thread(target=_runner, daemon=True)
    t.start()

_start_cleanup_thread_once()
logger.info("In-memory AI Interview Voicebot backend loaded")

# Export functions
__all__ = [
    'process_audio_pipeline_memory',
    'perf_monitor',
    'numpy_to_wav_bytes',
    'wav_bytes_to_numpy',
    'is_configured',
    'config_status',
    'process_audio_pipeline',  # Legacy compatibility
    'save_numpy_audio'         # Legacy compatibility
]
