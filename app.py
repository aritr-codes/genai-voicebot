import gradio as gr
from gradio.themes import Soft
import os
import logging
import traceback
import sys
from typing import Any
import numpy as np

# Setup logging for Spaces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import backend functions (soft-fail for UI startup)
BACKEND_OK = True
MISSING_REASON = ""
try:
    from backend import (
        process_audio_pipeline_memory,
        perf_monitor,
        numpy_to_wav_bytes,
        is_configured,
        config_status
    )
    logger.info("Backend imported successfully")
except Exception as e:
    BACKEND_OK = False
    MISSING_REASON = str(e)
    logger.warning(f"Backend not fully available: {e}")
    def is_configured() -> bool:
        return False
    def config_status() -> dict:
        return { 'OPENAI_API_KEY': False, 'ASSEMBLYAI_API_KEY': False, 'ELEVENLABS_API_KEY': False }
    def process_audio_pipeline_memory(*args: Any, **kwargs: Any) -> tuple:
        raise RuntimeError(f"Backend unavailable: {MISSING_REASON}")
    def numpy_to_wav_bytes(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError(f"Audio conversion unavailable: {MISSING_REASON}")
    class _Perf:
        def get_avg_times(self):
            return {}
    perf_monitor = _Perf()

# Soft-check environment/configuration (don't crash UI)
try:
    cfg = config_status()
    if not all(cfg.values()):
        logger.warning(f"Missing environment variables: {[k for k,v in cfg.items() if not v]}")
    else:
        logger.info("All required environment variables are present")
except Exception:
    logger.warning("Could not determine configuration status at import time")

# Custom CSS optimized for Spaces
custom_css = """
/* Spaces-optimized styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif !important;
}

.main {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin: 20px auto;
}

.title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    color: #1a202c;
    margin: 0 0 8px;
}

.subtitle {
    text-align: center;
    color: #4a5568;
    font-size: 16px;
    margin-bottom: 24px;
}

.panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}

.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
}

.primary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.status {
    text-align: center;
    color: #2d3748;
    font-size: 15px;
    margin: 12px 0;
    padding: 8px;
    border-radius: 6px;
}

.status.error {
    background: #fed7d7;
    color: #c53030;
}

.status.success {
    background: #c6f6d5;
    color: #2f855a;
}

.status.processing {
    background: #bee3f8;
    color: #2c5aa0;
}

/* Hide Gradio footer and branding */
footer { display: none !important; }
.gradio-container .footer { display: none !important; }
.gradio-container .version { display: none !important; }

/* Audio component styling */
.audio-container {
    background: white;
    border-radius: 8px;
    padding: 16px;
    border: 2px dashed #cbd5e0;
    transition: all 0.2s ease;
}

.audio-container:hover {
    border-color: #667eea;
}

/* Responsive design */
@media (max-width: 768px) {
    .main { padding: 16px; margin: 10px; }
    .title { font-size: 24px; }
    .panel { padding: 16px; }
}
"""

def _format_metrics():
    """Format performance metrics for display"""
    try:
        avg = perf_monitor.get_avg_times()
        if not avg:
            return "No metrics available yet."

        def ms(v: float) -> str:
            return f"{(v or 0) * 1000:.0f}ms"

        cache_rate = avg.get('cache_hit_rate', 0.0) * 100.0

        lines = [
            f"**Average Processing Times:**",
            f"• Transcription: {ms(avg.get('avg_transcription', 0.0))}",
            f"• LLM Generation: {ms(avg.get('avg_llm', 0.0))}",
            f"• Text-to-Speech: {ms(avg.get('avg_tts', 0.0))}",
            f"• **Total Pipeline: {ms(avg.get('avg_total', 0.0))}**",
            f"• Cache Hit Rate: {cache_rate:.1f}%"
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Error formatting metrics: {e}")
        return "Metrics unavailable"

## UI is defined after handlers below to allow click bindings inside Blocks

def handle_generate(audio_data):
    """
    Enhanced audio generation handler with better async handling.
    """
    logger.info(f"=== Starting audio processing ===")

    # Early yield to prevent UI blocking
    yield (
        '<div class="status processing">🎤 Receiving audio data...</div>',
        gr.update(visible=False),
        None, None, None,
        _format_metrics(),
        gr.update(value="⏳ Processing...", interactive=False)
    )

    # Debug audio data
    if audio_data is not None and isinstance(audio_data, (tuple, list)):
        sr, data = audio_data
        if hasattr(data, 'shape'):
            logger.info(f"Audio debug - Shape: {data.shape}, dtype: {data.dtype}, "
                    f"min: {np.min(data) if data.size > 0 else 'N/A'}, "
                    f"max: {np.max(data) if data.size > 0 else 'N/A'}, "
                    f"mean: {np.mean(np.abs(data)) if data.size > 0 else 'N/A'}")

    # Validate input
    if audio_data is None:
        logger.warning("No audio data received")
        yield (
            '<div class="status error">❌ Please record a question first.</div>',
            gr.update(visible=False),
            None, None, None,
            _format_metrics(),
            gr.update(value="🚀 Generate AI Response", interactive=True)
        )
        return

    try:
        # Check backend readiness
        if not BACKEND_OK or not is_configured():
            missing = []
            try:
                st = config_status()
                missing = [k for k,v in st.items() if not v]
            except Exception:
                pass
            msg = "Missing configuration. " + (f"Set {', '.join(missing)}." if missing else "Please set required API keys.")
            yield (
                f'<div class="status error">❌ {msg}</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )
            return

        # Validate audio data format
        if not isinstance(audio_data, (tuple, list)) or len(audio_data) != 2:
            logger.error(f"Invalid audio data format. Expected tuple/list of length 2, got: {type(audio_data)}")
            yield (
                '<div class="status error">❌ Invalid audio format. Please try recording again.</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )
            return

        sr, data = audio_data

        # Check for empty or very short recordings
        if data is None or (hasattr(data, 'size') and data.size == 0):
            logger.warning("Empty audio data received")
            yield (
                '<div class="status error">❌ Recording is empty. Please try again.</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )
            return

        # Log audio properties
        logger.info(f"Audio properties - Sample rate: {sr}, Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

        # Yield status before conversion
        yield (
            '<div class="status processing">🔄 Converting audio format...</div>',
            gr.update(visible=False),
            None, None, None,
            _format_metrics(),
            gr.update(value="⏳ Converting...", interactive=False)
        )

        # Convert numpy audio to WAV bytes for processing
        try:
            # Add a small delay to let the UI update
            import time
            time.sleep(0.1)

            wav_bytes = numpy_to_wav_bytes(audio_data)
            logger.info(f"Successfully converted to WAV: {len(wav_bytes)} bytes")

        except Exception as e:
            logger.error(f"WAV conversion failed: {e}")
            yield (
                f'<div class="status error">❌ Audio conversion failed: {str(e)}</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )
            return

        # Show processing status
        yield (
            '<div class="status processing">🎧 Processing your audio... This may take 15-30 seconds.</div>',
            gr.update(visible=False),
            None, None, None,
            _format_metrics(),
            gr.update(value="⏳ Processing...", interactive=False)
        )

        # Process through the in-memory pipeline
        try:
            logger.info("Starting in-memory pipeline...")
            transcript, numpy_audio, ai_response = process_audio_pipeline_memory(
                wav_bytes,
                voice='3gsg3cxXyFLcGIfNbM6C',
                speed=1.0
            )
            logger.info(f"Pipeline completed. Transcript preview: {transcript[:100] if transcript else 'None'}...")

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Pipeline processing failed: {e}\nTraceback: {error_trace}")
            yield (
                f'<div class="status error">❌ Processing failed: {str(e)}</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )
            return

        # Check results and provide appropriate feedback
        if transcript and not str(transcript).startswith('ERROR:'):
            logger.info("✅ Processing completed successfully")
            yield (
                '<div class="status success">✅ Response generated successfully!</div>',
                gr.update(visible=True),
                numpy_audio, ai_response, transcript,
                _format_metrics(),
                gr.update(value="🔄 Generate Another", interactive=True)
            )
        else:
            error_msg = transcript if (transcript and isinstance(transcript, str)) else "Unknown processing error occurred"
            logger.error(f"Pipeline returned error: {error_msg}")
            yield (
                f'<div class="status error">❌ {error_msg.replace("ERROR: ", "")}</div>',
                gr.update(visible=False),
                None, None, None,
                _format_metrics(),
                gr.update(value="🔄 Try Again", interactive=True)
            )

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in handle_generate: {e}\nTraceback: {error_trace}")
        yield (
            f'<div class="status error">❌ Unexpected error: {str(e)}</div>',
            gr.update(visible=False),
            None, None, None,
            _format_metrics(),
            gr.update(value="🔄 Try Again", interactive=True)
        )

def handle_clear():
    """Clear the interface and reset to initial state."""
    logger.info("Clearing interface")
    return (
        '<div class="status">Ready to process your question</div>',
        gr.update(visible=False),
        None,  # Clear audio input
        None,  # Clear AI audio
        None,  # Clear AI text
        _format_metrics()  # Refresh metrics
    )

## Build UI Blocks and wire events (must be inside context)
with gr.Blocks(
    title="AI Interview Assistant",
    theme=Soft(),
    css=custom_css,
    analytics_enabled=False
) as demo:
    with gr.Column(elem_classes=["main"]):
        gr.HTML('<div class="title">🎤 AI Interview Assistant</div>')
        gr.HTML('<div class="subtitle">Practice interviews with AI-powered responses • Optimized for Hugging Face Spaces</div>')

        with gr.Column(elem_classes=["panel"]):
            gr.HTML('<h3 style="margin:0 0 12px;color:#2d3748;">📝 Record Your Question</h3>')
            audio_input = gr.Audio(
                label="Click to record (max 10 seconds)",
                type="numpy",
                sources=["microphone"],
                max_length=10,
                show_label=True,
                interactive=True,
                elem_classes=["audio-container"]
            )
            gr.HTML('<p style="font-size:14px;color:#718096;margin:8px 0 0;text-align:center;">Speak clearly and wait for the recording to complete before processing.</p>')

        with gr.Row():
            generate_btn = gr.Button("🚀 Generate AI Response", elem_classes=["primary-btn"], size="lg", scale=2)
            clear_btn = gr.Button("🔄 Clear", size="lg", scale=1)

        status_html = gr.HTML('<div class="status">Ready to process your question</div>')

        with gr.Column(visible=False, elem_classes=["panel"]) as response_panel:
            gr.HTML('<h3 style="margin:0 0 16px;color:#2d3748;">🤖 AI Response</h3>')
            ai_audio = gr.Audio(label="🔊 Listen to Response", type="numpy", interactive=False, autoplay=True, show_download_button=True)
            with gr.Accordion("📄 View Text Details", open=False):
                ai_text = gr.Textbox(label="AI Response Text", lines=4, interactive=False, show_copy_button=True, placeholder="The AI response will appear here...")
                transcript_box = gr.Textbox(label="Your Question (Transcribed)", lines=2, interactive=False, show_copy_button=True, placeholder="Your speech will be transcribed here...")

        with gr.Accordion("📊 Performance Metrics", open=False):
            perf_text = gr.Markdown(_format_metrics())

        generate_btn.click(
            fn=handle_generate,
            inputs=[audio_input],
            outputs=[status_html, response_panel, ai_audio, ai_text, transcript_box, perf_text, generate_btn],
            queue=True,
            show_progress="full",
            api_name="generate_response"
        )

        clear_btn.click(
            fn=handle_clear,
            outputs=[status_html, response_panel, audio_input, ai_audio, ai_text, perf_text]
        )

        with gr.Row():
            gr.HTML('''
            <div style="text-align:center;color:#718096;font-size:14px;margin-top:20px;padding:16px;">
                <p><strong>💡 Tips for best results:</strong></p>
                <p>• Speak clearly and at normal pace • Keep questions under 30 seconds • Wait for processing to complete</p>
                <p><strong>🔧 Powered by:</strong> OpenAI GPT-3.5 • AssemblyAI • ElevenLabs • Optimized for Hugging Face Spaces</p>
            </div>
            ''')

# Configure and launch the application
if __name__ == "__main__":
    logger.info("🚀 Starting AI Interview Assistant...")

    try:
        # More aggressive queue configuration for audio processing
        demo.queue(
            max_size=10,                      # Smaller queue to prevent memory buildup
            default_concurrency_limit=1,      # Process one at a time to prevent overload
            status_update_rate="auto"         # More frequent status updates
        )

        # Launch with optimized configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=int(os.getenv("PORT", "7860")),
            show_error=True,
            ssr_mode=False,
            show_api=False,
            quiet=False,
            favicon_path=None,
            root_path=os.getenv("GRADIO_ROOT_PATH", ""),
            max_threads=40,              # Increase thread pool for better async handling
            app_kwargs={
                "docs_url": None,
                "redoc_url": None
            }
        )

    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
