import os
import cv2
import base64
import gc
from io import BytesIO
import gradio as gr
import numpy as np
import torch
from webcamgpt.core import Qwen2V_SAM2_Connector

MARKDOWN = """
# Qwen2V Image Segmentation Demo 💬 + 🖼️

This demo allows you to perform image segmentation using Qwen2V + SAM2.
Upload an image and enter your prompt to begin!
"""

def cleanup_memory():
    """Clean up memory and CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize image to specified size."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    cleanup_memory()
    return resized

def process_mask(mask_logit, target_size: tuple = (512, 512)) -> np.ndarray:
    """Process a single mask with memory management"""
    try:
        if isinstance(mask_logit, torch.Tensor):
            mask = mask_logit.cpu().numpy()
        else:
            mask = mask_logit
            
        mask = np.squeeze(mask)
        mask = (mask * 255).astype(np.uint8)
        
        if mask.shape != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
        return mask
    finally:
        if isinstance(mask_logit, torch.Tensor):
            del mask_logit
            cleanup_memory()

def combine_masks(masks: list, target_size: tuple = (512, 512)) -> np.ndarray:
    """Combine multiple masks into a single mask with memory management"""
    try:
        combined_mask = np.zeros(target_size, dtype=np.uint8)
        
        for mask_logit in masks:
            try:
                mask = process_mask(mask_logit, target_size)
                combined_mask = cv2.max(combined_mask, mask)
                del mask
                cleanup_memory()
            except Exception as e:
                print(f"Error processing individual mask: {e}")
                continue
        
        return combined_mask
    except Exception as e:
        print(f"Error in combine_masks: {e}")
        return np.zeros(target_size, dtype=np.uint8)
    finally:
        cleanup_memory()

def apply_colored_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply colored overlay to image using mask with memory management"""
    try:
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
        return overlay
    finally:
        cleanup_memory()

def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string with memory management"""
    try:
        _, buffer = cv2.imencode('.jpeg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    finally:
        cleanup_memory()

# Initialize Qwen model
model = Qwen2V_SAM2_Connector()

def respond(image: np.ndarray, prompt: str, chat_history):
    try:
        cleanup_memory()  # Initial cleanup
        
        if image is None:
            chat_history.append((prompt, "Please upload an image first."))
            return "", chat_history
        
        try:
            # Resize and preprocess image
            image = resize_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Generate masks based on prompt
            masks = model.process_image_text(image, prompt)
            
            if masks and len(masks) > 0:
                # Combine masks and apply overlay
                combined_mask = combine_masks(masks)
                overlay_image = apply_colored_overlay(image, combined_mask)
                
                # Convert result to base64
                img_base64 = image_to_base64(overlay_image)
                img_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
                
                chat_history.append((prompt, img_html))
                
                # Clean up intermediate results
                del combined_mask
                del overlay_image
                cleanup_memory()
            else:
                chat_history.append((prompt, "No masks detected."))
                
            response = "Segmentation completed. Check the image with masks."
            chat_history.append((None, response))
            
        except Exception as process_error:
            print(f"Error processing image: {process_error}")
            model.reinitialize()  # Reinitialize the model on error
            raise
        
        return "", chat_history
    
    except Exception as e:
        print(f"Error in respond function: {str(e)}")
        response = f"An error occurred while processing your request: {str(e)}"
        chat_history.append((prompt, response))
        cleanup_memory()
        return "", chat_history
    
    finally:
        cleanup_memory()

def on_clear():
    """Cleanup function for clear button"""
    cleanup_memory()
    model.reinitialize()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image")
        with gr.Column():
            chatbot = gr.Chatbot(height=500, allow_html=True)
            message = gr.Textbox(label="Enter your prompt")
            clear_button = gr.ClearButton([message, chatbot])
    
    # Add cleanup on clear
    clear_button.click(on_clear)
    
    # Submit handler
    message.submit(
        respond,
        inputs=[image_input, message, chatbot],
        outputs=[message, chatbot]
    )

if __name__ == "__main__":
    demo.launch(debug=False, show_error=True)