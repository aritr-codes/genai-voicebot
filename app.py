import gradio as gr
from gradio.themes import Soft

# Import robust backend pipeline and metrics
from backend import process_audio_pipeline, perf_monitor, save_numpy_audio


custom_css = """
/* Minimal, clean styling focused on UX */
.gradio-container { max-width: 1220px !important; margin: 0 auto !important; }
.gradio-container, body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, 'Noto Sans', 'Helvetica Neue', Arial, sans-serif !important; }
.main { background: #ffffff; border-radius: 20px; padding: 28px; box-shadow: 0 12px 36px rgba(0,0,0,0.08); }
.title { text-align: center; font-size: 30px; font-weight: 700; color: #2d3748; margin: 6px 0 8px; }
.subtitle { text-align: center; color: #718096; font-size: 16px; margin-bottom: 28px; }
.panel { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; }
.primary { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color: #fff; border: none; font-weight: 600; border-radius: 10px; }
.status { text-align:center; color:#4a5568; font-size: 15px; margin: 10px 0 0; }
footer { display:none !important; }
"""


def _format_metrics():
    avg = perf_monitor.get_avg_times()
    if not avg:
        return "No metrics yet. Run at least one request."
    def ms(v: float) -> str:
        return f"{(v or 0) * 1000:.0f} ms"
    cache_rate = avg.get('cache_hit_rate', 0.0) * 100.0
    lines = [
        f"Transcription avg: {ms(avg.get('avg_transcription', 0.0))}",
        f"LLM avg: {ms(avg.get('avg_llm', 0.0))}",
        f"TTS avg: {ms(avg.get('avg_tts', 0.0))}",
        f"Total avg: {ms(avg.get('avg_total', 0.0))}",
        f"Cache hit rate: {cache_rate:.1f}%",
    ]
    return "\n".join(lines)


with gr.Blocks(title="AI Interview Assistant", theme=Soft(), css=custom_css) as demo:
    with gr.Column(elem_classes=["main"]):
        gr.HTML('<div class="title">AI Interview Assistant</div>')
        gr.HTML('<div class="subtitle">Speak a question, get a concise, professional answer</div>')

        with gr.Column(elem_classes=["panel"]):
            audio_input = gr.Audio(
                label="🎙️ Record your question (max ~30s)",
                # type="filepath",
                type="numpy",     # CHANGED: numpy instead of filepath
                sources=["microphone"],
                max_length=30,
                show_label=True,
                interactive=True,
            )

        generate_btn = gr.Button("Generate Interview Response", elem_classes=["primary"], size="lg")
        status_html = gr.HTML('<div class="status">Ready</div>')

        with gr.Column(visible=False, elem_classes=["panel"]) as response_panel:
            gr.HTML('<h3 style="margin:0 0 10px;color:#4a5568;">Your AI Response</h3>')
            ai_audio = gr.Audio(label="🔊 Play response", type="filepath", interactive=False, autoplay=True)
            with gr.Accordion("📝 View Response Text", open=False):
                ai_text = gr.Textbox(label="Response", lines=4, interactive=False, show_copy_button=True)
                transcript_box = gr.Textbox(label="Your Question (Transcript)", lines=2, interactive=False, show_copy_button=True)

        with gr.Accordion("📈 Performance (averages)", open=False):
            perf_text = gr.Markdown(_format_metrics())

        with gr.Row():
            clear_btn = gr.Button("🔄 Start Over")

    # def handle_generate(audio_file):
    #     if not audio_file:
    #         return (
    #             '<div class="status" style="color:#e53e3e;">Please record a question first.</div>',
    #             gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Generate Interview Response", interactive=True)
    #         )

    #     # progress update
    #     yield (
    #         '<div class="status">🎧 Processing your audio... This may take a few seconds.</div>',
    #         gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Generating...", interactive=False)
    #     )

    #     # Use provided ElevenLabs voice ID for Raju
    #     transcript, wav_path, ai_response = process_audio_pipeline(audio_file, voice='3gsg3cxXyFLcGIfNbM6C', speed=1.0)

    #     if transcript and not str(transcript).startswith('ERROR:'):
    #         # success
    #         yield (
    #             '<div class="status" style="color:#38a169;">✅ Response generated successfully!</div>',
    #             gr.update(visible=True), wav_path, ai_response, transcript, _format_metrics(), gr.update(value="Generate Another", interactive=True)
    #         )
    #     else:
    #         # error
    #         error_msg = transcript if transcript else "Unknown error occurred"
    #         yield (
    #             f'<div class="status" style="color:#e53e3e;">❌ {error_msg}</div>',
    #             gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Try Again", interactive=True)
    #         )

    def handle_generate(audio_data):
        if audio_data is None:
            return (
                '<div class="status" style="color:#e53e3e;">Please record a question first.</div>',
                gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Generate Interview Response", interactive=True)
            )

        # progress update
        yield (
            '<div class="status">🎧 Processing your audio... This may take a few seconds.</div>',
            gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Generating...", interactive=False)
        )

        # ✅ FIX: convert numpy -> wav path first
        audio_path = save_numpy_audio(audio_data)

        # Now pass file path into pipeline
        transcript, wav_path, ai_response = process_audio_pipeline(audio_path, voice='3gsg3cxXyFLcGIfNbM6C', speed=1.0)

        if transcript and not str(transcript).startswith('ERROR:'):
            yield (
                '<div class="status" style="color:#38a169;">✅ Response generated successfully!</div>',
                gr.update(visible=True), wav_path, ai_response, transcript, _format_metrics(), gr.update(value="Generate Another", interactive=True)
            )
        else:
            error_msg = transcript if transcript else "Unknown error occurred"
            yield (
                f'<div class="status" style="color:#e53e3e;">❌ {error_msg}</div>',
                gr.update(visible=False), None, None, None, _format_metrics(), gr.update(value="Try Again", interactive=True)
            )

    def handle_clear():
        return (
            '<div class="status">Ready</div>',
            gr.update(visible=False), None, None, None, _format_metrics()
        )

    generate_btn.click(
        fn=handle_generate,
        inputs=[audio_input],
        outputs=[status_html, response_panel, ai_audio, ai_text, transcript_box, perf_text, generate_btn],
        queue=True,
        api_name="generate",
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[status_html, response_panel, audio_input, ai_audio, ai_text, perf_text],
    )


import os

if __name__ == "__main__":
    demo.queue(max_size=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        ssr_mode=False,
    )
