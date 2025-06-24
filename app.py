import os
import gradio as gr
import shutil
from inference import tango
import numpy as np


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


examples_audio = [
    ["./datasets/cached_audio/example_male_voice_9_seconds.wav"],
    # ["./datasets/cached_audio/example_female_voice_9_seconds.wav"],
]

examples_video = [
    # ["./datasets/cached_audio/speaker8_jjRWaMCWs44_00-00-30.16_00-00-33.32.mp4"],
    # ["./datasets/cached_audio/speaker7_iuYlGRnC7J8_00-00-0.00_00-00-3.25.mp4"],
    ["./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4"],
    # ["./datasets/cached_audio/1wrQ6Msp7wM_00-00-39.69_00-00-45.68.mp4"],
    # ["./datasets/cached_audio/101099-00_18_09-00_18_19.mp4"],
]

combined_examples = [
    ["./datasets/cached_audio/example_female_voice_9_seconds.wav", "./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4", 2024],
]


def tango_wrapper(audio_path, character_name, seed=2024, create_graph=False, video_folder_path=None):
    if isinstance(audio_path, tuple):
        sample_rate, audio_waveform = audio_path
        if audio_waveform.dtype != np.float32:
            audio_waveform = audio_waveform.astype(np.float32) / 32768.0
        audio_path = (sample_rate, audio_waveform)
    return tango(audio_path, character_name, seed=seed, create_graph=create_graph, video_folder_path=video_folder_path)


def make_demo():
    with gr.Blocks(analytics_enabled=False) as Interface:
        gr.Markdown(
            """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
          <div>
            <h1>TANGO</h1>
            <span>Generating full-body talking videos from audio and reference video</span>
            <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
              <a href='https://h-liu1997.github.io/'>Haiyang Liu</a>, \
              <a href='https://yangxingchao.github.io/'>Xingchao Yang</a>, \
              <a href=''>Tomoya Akiyama</a>, \
              <a href='https://sky24h.github.io/'> Yuantian Huang</a>, \
              <a href=''>Qiaoge Li</a>, \
              <a href='https://www.tut.ac.jp/english/university/faculty/cs/164.html'>Shigeru Kuriyama</a>, \
              <a href='https://taketomitakafumi.sakura.ne.jp/web/en/'>Takafumi Taketomi</a>\
            </h2>
            <br>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
              <a href="https://arxiv.org/abs/2410.04221"><img src="https://img.shields.io/badge/arXiv-2410.04221-blue"></a>
              &nbsp;
              <a href="https://pantomatrix.github.io/TANGO/"><img src="https://img.shields.io/badge/Project_Page-TANGO-orange" alt="Project Page"></a>
              &nbsp;
              <a href="https://github.com/CyberAgentAILab/TANGO"><img src="https://img.shields.io/badge/Github-Code-green"></a>
              &nbsp;
              <a href="https://github.com/CyberAgentAILab/TANGO"><img src="https://img.shields.io/github/stars/CyberAgentAILab/TANGO
              "></a>
            </div>
          </div>
        </div>
        """
        )

        # Create a gallery with 5 videos
        with gr.Row():
            gr.Video(value="./datasets/cached_audio/demo1.mp4", label="Demo 0")
            gr.Video(value="./datasets/cached_audio/demo2.mp4", label="Demo 1")
            gr.Video(value="./datasets/cached_audio/demo3.mp4", label="Demo 2")
            gr.Video(value="./datasets/cached_audio/demo4.mp4", label="Demo 3")
            gr.Video(value="./datasets/cached_audio/demo5.mp4", label="Demo 4")
        with gr.Row():
            gr.Video(value="./datasets/cached_audio/demo6.mp4", label="Demo 5")
            gr.Video(value="./datasets/cached_audio/demo0.mp4", label="Demo 6")
            gr.Video(value="./datasets/cached_audio/demo7.mp4", label="Demo 7")
            gr.Video(value="./datasets/cached_audio/demo8.mp4", label="Demo 8")
            gr.Video(value="./datasets/cached_audio/demo9.mp4", label="Demo 9")

        with gr.Row():
            gr.Markdown(
                """
              <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
              This is an open-source project running locally, operates in low-quality mode. Some generated results from high-quality mode are shown above.
              <br>
              News:
              <br>
              [10/15]: Add watermark, fix bugs on custom character by downgrades to py3.9, fix bugs to support audio less than 4s.
              </div>
              """
            )

        with gr.Row():
            with gr.Column(scale=4):
                video_output_1 = gr.Video(
                    label="Generated video - 1",
                    interactive=False,
                    autoplay=False,
                    loop=False,
                    show_share_button=True,
                )
            with gr.Column(scale=4):
                video_output_2 = gr.Video(
                    label="Generated video - 2",
                    interactive=False,
                    autoplay=False,
                    loop=False,
                    show_share_button=True,
                )
            with gr.Column(scale=1):
                file_output_1 = gr.File(label="Download 3D Motion and Visualize in Blender")
                file_output_2 = gr.File(label="Download 3D Motion and Visualize in Blender")
                gr.Markdown("""
                <div style="display: flex; justify-content: center; align-items: center; text-align: left;">
                Details of the low-quality mode:
                <br>
                1. lower resolution, video resized as long-side 512 and keep aspect ratio.
                <br>
                2. subgraph instead of full-graph, causing noticeable "frame jumps". 
                <br>
                3. only use the first 8s of your input audio.
                <br>
                4. only use the first 20s of your input video for custom character. if you custom character, it will only generate one video result without "smoothing" for saving time.
                <br>
                5. use open-source tools like SMPLerX-s-model, Wav2Lip, and FiLM for faster processing. 
                <br>
                <br>
                Feel free to open an issue on GitHub or contact the authors if this does not meet your needs.
                </div>
                """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload your audio")
                seed_input = gr.Number(label="Seed", value=2024, interactive=True)
            with gr.Column(scale=2):
                gr.Examples(
                    examples=examples_audio,
                    inputs=[audio_input],
                    outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
                    label="Select existing Audio examples",
                    cache_examples=False,
                )
            with gr.Column(scale=1):
                video_input = gr.Video(label="Default Character", value="./datasets/cached_audio/speaker9_o7Ik1OB4TaE_00-00-38.15_00-00-42.33.mp4", interactive=False, elem_classes="video")
                gr.Markdown(
                    """
                    Custom character upload is not supported in gradio 5.x (python 3.10).
                    <br>
                    To use it, download to local and set up a py39 environment for SimplerX and mmcv.
                    """
                )
            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    The character is fixed to the default one on the left.
                    """
                )

        # Fourth row: Generate video button
        with gr.Row():
            run_button = gr.Button("Generate Video")

        # Define button click behavior
        run_button.click(
            fn=tango_wrapper,
            inputs=[audio_input, video_input, seed_input],
            outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
        )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Examples(
                    examples=combined_examples,
                    inputs=[audio_input, video_input, seed_input],  # Both audio and video as inputs
                    outputs=[video_output_1, video_output_2, file_output_1, file_output_2],
                    fn=tango_wrapper,  # Function that processes both audio and video inputs
                    label="Select Combined Audio and Video Examples (Cached)",
                    cache_examples=True,
                )

    return Interface


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8675"

    demo = make_demo()
    demo.launch(share=True)
