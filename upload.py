import streamlit as st

import subprocess
import os
from moviepy.editor import *
from preprocess import *
from llod_chat import *
import time
import RT


# Function to process video with object detection
def process_video(input_video_path, weights):
    command = f"python E:\\LLODProj1\\yolov5\\detect.py --weights {weights} --source {input_video_path} --save-txt"
    process = subprocess.Popen(command, shell=True)
    process.communicate()


def uploader(temp_video_path):
    try:
        start_time = time.time()
        with open(temp_video_path, "rb") as f:
            video_bytes = f.read()
            st.video(video_bytes, format="video/mp4", start_time=0)
        return start_time
    except Exception:
        return 0


# Search for all folders with names starting with 'exp' in the specified root folder
def find_latest_exp_folder(root_folder):
    exp_folders = glob.glob(os.path.join(root_folder, "exp*"))
    exp_folders.sort(key=os.path.getmtime, reverse=True)
    if exp_folders:
        return exp_folders[0]
    else:
        return None


txt_folder_path = os.path.join(
    find_latest_exp_folder("E:\\LLODProj1\\yolov5\\runs\\detect"), "labels"
)


def enhancer(temp_video_path):
    try:
        enhanced_clip = enhance_video(temp_video_path)
        enhanced_clip.close()
        enhanced_video_path = "E:\\LLODProj1\\temp\\enhanced.mp4"
        enhanced_clip.write_videofile(enhanced_video_path, codec="libx264", audio=False)

        # Check if enhanced video file was created successfully
        if not os.path.isfile(enhanced_video_path):
            raise FileNotFoundError(
                f"Enhanced video file not found: {enhanced_video_path}"
            )

        denoised_video_path = "E:\\LLODProj1\\temp\\denoised.mp4"
        denoise_video(enhanced_video_path, denoised_video_path)

        # Check if denoised video file was created successfully
        if not os.path.isfile(denoised_video_path):
            raise FileNotFoundError(
                f"Denoised video file not found: {denoised_video_path}"
            )

        with open(enhanced_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes, format="video/mp4", start_time=0)  # Update format here
        os.remove(enhanced_video_path)
        return denoised_video_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return 0


def detector(denoised_video_path, weights):
    try:
        process_video(denoised_video_path, weights)
        exp_folder = find_latest_exp_folder("E:\\LLODProj1\\yolov5\\runs\\detect")
        if exp_folder is not None:
            output_video_path = os.path.join(exp_folder, "denoised.mp4")
            clip = VideoFileClip(output_video_path)
            result_video_path = "E:\\LLODProj1\\output\\result.mp4"
            clip.write_videofile(result_video_path, codec="libx264")
            with open(result_video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes, format="video/mp4", start_time=0)
        end_time = time.time()
        return end_time
    except Exception:
        return 0


def main():
    st.set_page_config(page_title="LLOD", layout="wide")
    st.markdown(
        """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    .st-emotion-cache-keje6w  {
        position:relative;
        left: 5.8vw;
    }
    .st-emotion-cache-keje6w > div {
        margin:1.5vw;
    }
        """,
        unsafe_allow_html=True,
    )
    # Upload Model Options
    sidebar_title = """<p style="font-family: Source Sans Pro; color:White; 
                    font-size: 33px; margin-bottom:3vh;">OPTIONS</p>"""
    st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)
    weight_options1 = {
        "Fast (Less accurate)": "E:\\LLODProj1\\yolov5s6.pt",
        "Balanced (Moderate accuracy)": "E:\\LLODProj1\\yolov5m6.pt",
        "Accurate (Slower, best accuracy)": "E:\\LLODProj1\\yolov5x6.pt",
    }
    selected_weight1 = st.sidebar.radio(
        "**Upload Detection**", list(weight_options1.keys())
    )
    # Real Time Model Options
    weight_options2 = {
        "Fast detection": "E:\\LLODProj1\\yolov8n.pt",
        "Balanced detection": "E:\\LLODProj1\\yolov8s.pt",
        "Accurate detection": "E:\\LLODProj1\\yolov8m.pt",
    }
    selected_weight2 = st.sidebar.radio(
        "**Real-time Detection**", list(weight_options2.keys())
    )
    # Body
    page_title = """<p style= "font-family: Source Sans Pro; 
                    color:White; font-size: 45px; text-align:center;
                    position: relative; bottom: 5vh; margin-bottom: 2vh;">LOW LIGHT OBJECT DETECTION</p>"""
    st.markdown(page_title, unsafe_allow_html=True)

    subcol4, subcol5 = st.columns((1, 1))
    with subcol4:
        st.markdown(
            "<h2 style = margin: auto; padding: auto; text-align: center;> Upload Video</h2>",
            unsafe_allow_html=True,
        )
        st.text(f"{selected_weight1} weights selected")
        # File uploader in the first column
        uploaded_file = st.file_uploader("", type=["mp4"])

        if uploaded_file is not None:
            temp_video_path = os.path.join("E:\\LLODProj1", "temp", uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())
                # Upload
                st.write("Uploading file...")
                start_time = uploader(temp_video_path)
                if start_time != 0:
                    st.write("File uploaded successfully!")
                else:
                    st.write("An error occurred. Please try again.")
                st.write("Now processing the video...")
                denoised_video_path = enhancer(temp_video_path)
                if denoised_video_path != 0:
                    st.write("Video enhancement completed!")
                else:
                    st.write("Could not process video. Please try again.")
                # Detect
                st.write(f"Now detecting objects")
                end_time = detector(
                    denoised_video_path, weight_options1[selected_weight1]
                )
                if end_time != 0:
                    st.write("Object detection completed!")
                    st.sidebar.title("Additional Info:")
                    total_time = end_time - start_time
                    st.sidebar.subheader(f"Total time taken: {total_time:.2f} seconds")
                    info = list()
                    if selected_weight1 == "Fast (Less accurate)":
                        info = [
                            "Number of detectable classes : 80",
                            "Depth : 0.33",
                            "Number of channels : 0.50",
                            "Weights file : yolov5s6.pt",
                            "Number of layers : 280",
                            "Number of parameters : 12612508",
                            "Gradients : 0",
                            "GFLOPs : 16.8",
                        ]
                    if selected_weight1 == "Accurate (Slower, best accuracy)":
                        st.sidebar.rows
                        info = [
                            "Number of detectable classes : 80",
                            "Depth : 0.33",
                            "Number of channels : 0.50",
                            "Weights file : yolov5x6.pt",
                            "Number of layers : 280",
                            "Number of parameters : 12612508",
                            "Gradients : 0",
                            "GFLOPs : 16.8",
                        ]

                    if selected_weight1 == "Balanced (Moderate accuracy)":
                        info = [
                            "Number of detectable classes : 80",
                            "Depth : 0.33",
                            "Number of channels : 0.50",
                            "Weights file : yolov5m6.pt",
                            "Number of layers : 280",
                            "Number of parameters : 12612508",
                            "Gradients : 0",
                            "GFLOPs : 16.8",
                        ]
                    markdown_list = "\n".join(
                        [f"{i+1}. {item}" for i, item in enumerate(info)]
                    )
                    # Display the information in the sidebar
                    st.sidebar.title("Model Information")
                    st.sidebar.markdown(markdown_list)
                else:
                    st.write("Detection failed. Please try again.")
            os.remove(temp_video_path)
            os.remove(denoised_video_path)
    with subcol5:
        st.markdown(
            "<h2 style = margin: auto; padding: auto; text-align: center;> Real Time Detection</h2>",
            unsafe_allow_html=True,
        )
        st.text(f"{selected_weight2} weights selected")
        yolov8_checkbox = st.checkbox("YOLOv8")
        if yolov8_checkbox:
            RT.rt_yolov8(weight_options2[selected_weight2])
        st.download_button(
            label="Download Detection Results",
            data="",
            file_name="data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
