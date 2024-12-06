import os
import cv2
import base64
import gradio as gr
import numpy as np
import torch
import gc
from algorithm.core import Qwen2V_SAM2_Connector

MARKDOWN = """
# Webcam Demo with Qwen2-VL 💬 + 📸

This demo allows you to chat with video using Qwen2-VL model.
"""

# Create Qwen2V_SAM2_Connector instance
connector = Qwen2V_SAM2_Connector()

def cleanup_memory():
    """Clean up memory after processing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def image_to_base64(image: np.ndarray) -> str:
    """Convert image to Base64 encoding"""
    _, buffer = cv2.imencode('.jpeg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def process_mask(mask_logit):
    """Process a single mask with memory management"""
    try:
        # Convert tensor to numpy if needed
        if isinstance(mask_logit, torch.Tensor):
            mask = mask_logit.cpu().numpy()
        else:
            mask = mask_logit
            
        mask = np.squeeze(mask)
        mask = (mask * 255).astype(np.uint8)
        return mask
        
    finally:
        # Clean up intermediate tensors
        if isinstance(mask_logit, torch.Tensor):
            del mask_logit
            cleanup_memory()

def respond(image: np.ndarray, prompt: str, chat_history):
    try:
        # Clear memory before processing
        cleanup_memory()
        
        # Preprocess image
        image = np.fliplr(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get masks using Qwen model
        masks = connector.process_image_text(image, prompt)
        
        # Visualize masks
        if masks and len(masks) > 0:
            overlay = image.copy()
            
            # Process masks one by one to manage memory
            for obj_id, mask_logit in enumerate(masks):
                try:
                    mask = process_mask(mask_logit)
                    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)
                    
                    # Clean up intermediate results
                    del mask
                    del colored_mask
                    cleanup_memory()
                    
                except Exception as mask_error:
                    print(f"Error processing mask {obj_id}: {mask_error}")
                    continue
            
            # Convert result to base64
            img_base64 = image_to_base64(overlay)
            img_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
            
            chat_history.append((prompt, img_html))
            
            # Clean up overlay
            del overlay
            
        else:
            chat_history.append((prompt, "No masks detected."))
        
        response = "Segmentation completed. Check the image with masks."
        chat_history.append((None, response))
        
        # Final cleanup
        cleanup_memory()
        
        return "", chat_history

    except Exception as e:
        print(f"Error in respond function: {str(e)}")
        response = f"An error occurred while processing your request: {str(e)}"
        chat_history.append((prompt, response))
        
        # Cleanup on error
        cleanup_memory()
        
        # Reinitialize connector on error
        connector.reinitialize()
        
        return "", chat_history
    
    finally:
        # Ensure memory is cleaned up
        cleanup_memory()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    
    with gr.Row():
        webcam = gr.Image(source="webcam", streaming=True, shape=(512, 512))
        with gr.Column():
            chatbot = gr.Chatbot(height=500, allow_html=True)
            message = gr.Textbox(placeholder="Enter your prompt here...")
            clear_button = gr.ClearButton([message, chatbot])

    # Add cleanup on clear
    def on_clear():
        cleanup_memory()
        connector.reinitialize()
    
    clear_button.click(on_clear)

    message.submit(
        respond,
        inputs=[webcam, message, chatbot],
        outputs=[message, chatbot]
    )

if __name__ == "__main__":
    demo.launch(debug=False, show_error=True)
