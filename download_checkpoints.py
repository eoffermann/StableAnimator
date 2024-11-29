import os
import requests
from bs4 import BeautifulSoup

# Define file structure and download links
download_structure = {
    "checkpoints/DWPose": {
        "files": {
            "dw-ll_ucoco_384.onnx": "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/DWPose/dw-ll_ucoco_384.onnx",
            "yolox_l.onnx": "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/DWPose/yolox_l.onnx",
        }
    },
    "checkpoints/Animation": {
        "files": {
            "pose_net.pth": "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/pose_net.pth",
            "face_encoder.pth": "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/face_encoder.pth",
            "unet.pth": "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/unet.pth",
        }
    },
    "checkpoints/SVD/stable-video-diffusion-img2vid-xt": {
        "directories": {
            "feature_extractor": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main/feature_extractor",
            "image_encoder": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main/image_encoder",
            "scheduler": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main/scheduler",
            "unet": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main/unet",
            "vae": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/main/vae",
        },
        "files": {
            "model_index.json": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/model_index.json",
            "svd_xt.safetensors": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors",
            "svd_xt_image_decoder.safetensors": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors",
        }
    }
}

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists. If it doesn't, create it.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def download_file(url, save_path):
    """
    Download a file from a URL and save it to a specified location.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")

def download_directory(url, save_directory):
    """
    Scrape a Hugging Face directory URL and download all files to the specified directory.
    """
    try:
        print(f"download_directory('{url}','{save_directory}')")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for file links in the page
        links = soup.find_all('a', class_='group flex items-center truncate')
        for link in links:
            href = link.get('href')
            if href and '/blob/' in href:
                href=href.replace('/blob/','/resolve/')
            if href and '/resolve/' in href:
                file_url = f"https://huggingface.co{href}"
                print(f"download file_url: {file_url}")
                file_name = href.split('/')[-1]
                save_path = os.path.join(save_directory, file_name)
                if not os.path.exists(save_path):
                    download_file(file_url, save_path)
                else:
                    print(f"File already exists: {save_path}")
    except Exception as e:
        print(f"Failed to download directory from {url}. Error: {e}")

def setup_checkpoints(download_structure):
    """
    Create directory structure and download files based on the given structure.
    """
    for directory, contents in download_structure.items():
        ensure_directory_exists(directory)
        # Handle file downloads
        for file_name, url in contents.get("files", {}).items():
            save_path = os.path.join(directory, file_name)
            if not os.path.exists(save_path):
                download_file(url, save_path)
            else:
                print(f"File already exists: {save_path}")
        # Handle directory downloads
        for sub_dir, url in contents.get("directories", {}).items():
            save_sub_dir = os.path.join(directory, sub_dir)
            ensure_directory_exists(save_sub_dir)
            download_directory(url, save_sub_dir)

if __name__ == "__main__":
    setup_checkpoints(download_structure)
