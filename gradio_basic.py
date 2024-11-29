import gradio as gr
import os
import torch
import numpy as np
from PIL import Image
from inference_basic import seed_everything, load_images_from_folder, save_frames_as_png, export_to_gif
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.modules.pose_net import PoseNet
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline

def run_inference(
    pretrained_model_name_or_path, 
    validation_image_path, 
    validation_control_folder, 
    output_dir, 
    height, 
    width, 
    guidance_scale, 
    num_inference_steps, 
    posenet_model_path, 
    face_encoder_model_path, 
    unet_model_path, 
    tile_size, 
    frames_overlap, 
    decode_chunk_size, 
    noise_aug_strength,
    seed
):
    # Initialize models
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(cross_attention_dim=1024, id_embeddings_dim=512, clip_embeddings_dim=1024, num_tokens=4)
    face_model = FaceModel()

    # Load pre-trained weights
    pose_net.load_state_dict(torch.load(posenet_model_path))
    face_encoder.load_state_dict(torch.load(face_encoder_model_path))
    unet.load_state_dict(torch.load(unet_model_path))

    # Prepare pipeline
    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda')

    # Load images
    validation_image = Image.open(validation_image_path).convert('RGB')
    control_images = load_images_from_folder(validation_control_folder, width, height)
    num_frames = len(control_images)

    # Run inference
    output_frames = pipeline(
        image=validation_image,
        image_pose=control_images,
        height=height,
        width=width,
        num_frames=num_frames,
        tile_size=tile_size,
        tile_overlap=frames_overlap,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=127.,
        fps=7,
        min_guidance_scale=guidance_scale,
        max_guidance_scale=guidance_scale,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
    ).frames

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    save_frames_as_png(output_frames, output_dir)
    export_to_gif(output_frames, os.path.join(output_dir, "animation_video.mp4"), fps=8)
    return os.path.join(output_dir, "animation_video.mp4")

# Gradio Interface
iface = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Textbox(label="Pretrained Model Path", value="checkpoints/SVD/stable-video-diffusion-img2vid-xt"),
        gr.Textbox(label="Validation Image Path", value="inference/case-1/reference.png"),
        gr.Textbox(label="Validation Control Folder", value="inference/case-1/poses"),
        gr.Textbox(label="Output Directory", value="basic_infer"),
        gr.Number(label="Height", value=1024),
        gr.Number(label="Width", value=576),
        gr.Number(label="Guidance Scale", value=3.0),
        gr.Number(label="Number of Inference Steps", value=25),
        gr.Textbox(label="PoseNet Model Path", value="checkpoints/Animation/pose_net.pth"),
        gr.Textbox(label="Face Encoder Model Path", value="checkpoints/Animation/face_encoder.pth"),
        gr.Textbox(label="UNet Model Path", value="checkpoints/Animation/unet.pth"),
        gr.Number(label="Tile Size", value=16),
        gr.Number(label="Frames Overlap", value=4),
        gr.Number(label="Decode Chunk Size", value=4),
        gr.Number(label="Noise Augmentation Strength", value=0.02),
        gr.Number(label="Seed", value=23123134),
    ],
    outputs=gr.Textbox(label="Output Video Path"),
    title="StableAnimator Gradio Interface",
    description="Run inference with StableAnimator via a user-friendly Gradio interface."
)

iface.launch()
