import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import sys
from hydra import initialize_config_dir
from omegaconf import OmegaConf
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
from openai import OpenAI
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from .utils import encode_image_to_base64
import gc

class OpenAI_SAM2_Connector:
    def __init__(self, api_key: Optional[str] = None):
        # OpenAI setup
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either through constructor or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        # SAM2 initialization
        SAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if SAM2_DIR not in sys.path:
            sys.path.append(SAM2_DIR)

        sam2_checkpoint = os.path.abspath(os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_base_plus.pt"))
        model_cfg = os.path.abspath(os.path.join(SAM2_DIR, "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"))

        config_dir = os.path.abspath(os.path.dirname(model_cfg))
        config_name = os.path.basename(model_cfg)

        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir):
            self.predictor = build_sam2_camera_predictor(
                config_name,
                sam2_checkpoint,
                device=self.device
            )

        self.ann_obj_id = 1

    def get_normalized_boxes(self, response_text: str) -> List[List[float]]:
        """Extract normalized coordinates from model response"""
        pattern = r'\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]'
        matches = re.findall(pattern, response_text)
        
        boxes = []
        for match in matches:
            try:
                box = [float(coord) for coord in match]
                if all(0 <= coord <= 1.0 for coord in box):
                    boxes.append(box)
            except ValueError:
                continue
        
        return boxes

    def get_pixel_boxes(self, normalized_boxes: List[List[float]], image_shape: Tuple[int, ...]) -> List[List[int]]:
        """Convert normalized coordinates to pixel coordinates"""
        height, width = image_shape[:2]
        pixel_boxes = []
        
        for box in normalized_boxes:
            x_min, y_min, x_max, y_max = box
            pixel_box = [
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height)
            ]
            pixel_boxes.append(pixel_box)
        
        return pixel_boxes

    def get_model_response(self, image: np.ndarray, prompt: str) -> str:
        """Get response from OpenAI GPT-4V model"""
        try:
            base64_image = encode_image_to_base64(image)
            
            system_prompt = """YPlease detect objects in this image and return their coordinates in normalized format between 0 and 1. Format: [[x_min, y_min, x_max, y_max], ...]. Be precise and accurate with coordinates.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in get_model_response: {str(e)}")
            raise

    def process_single_box(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """Generate mask for a single box"""
        try:
            # Transform box to SAM2 format (1, 2, 2)
            box_array = np.array(box).reshape(1, 2, 2)
            
            _, _, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.ann_obj_id,
                bbox=box_array
            )
            
            if out_mask_logits is not None and len(out_mask_logits) > 0:
                return (out_mask_logits[0] > 0.0).cpu().numpy()
            return None
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            return None

    def process_image_text(self, image: np.ndarray, prompt: str) -> List[np.ndarray]:
        """Process image and text to generate masks"""
        try:
            print(f"Processing image with shape: {image.shape}")
            
            # Get model response
            response = self.get_model_response(image, prompt)
            print(f"Model response: {response}")
            
            # Extract normalized coordinates
            normalized_boxes = self.get_normalized_boxes(response)
            if not normalized_boxes:
                print("No valid boxes found in response")
                return []
            
            # Convert to pixel coordinates
            pixel_boxes = self.get_pixel_boxes(normalized_boxes, image.shape)
            print(f"Found {len(pixel_boxes)} valid boxes")
            
            # Initialize SAM2
            self.predictor.load_first_frame(image)
            
            # Generate masks for each box
            masks = []
            for box in pixel_boxes:
                mask = self.process_single_box(image, box)
                if mask is not None:
                    masks.append(mask)
            
            print(f"Successfully generated {len(masks)} masks")
            return masks
            
        except Exception as e:
            print(f"Error in process_image_text: {str(e)}")
            return []

    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple masks into one"""
        if not masks:
            return None
        
        combined = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined = combined | mask
        return combined

    def track(self, image: np.ndarray) -> tuple:
        """Track objects in image"""
        try:
            self.predictor.add_conditioning_frame(image)
            obj_ids, masks = self.predictor.track(image)
            return obj_ids, masks
        except Exception as e:
            print(f"Error in tracking: {str(e)}")
            return [], []
        
        
import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import sys
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import gc

class Qwen2V_SAM2_Connector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Qwen2V_SAM2_Connector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        self.initialize_models()
        self.initialized = True

    def cleanup(self):
        """Clean up resources and free memory more aggressively"""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()  # Move model to CPU first
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if hasattr(self, 'predictor'):
                if hasattr(self.predictor, 'model'):
                    self.predictor.model.cpu()  # Move SAM model to CPU
                del self.predictor

            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            self.initialized = False
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")

    def initialize_models(self):
        """Initialize models with improved memory management"""
        # Clear existing resources first
        self.cleanup()
        
        try:
            # Qwen2-VL model load
            self.model_name = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map={'': self.device},
                attn_implementation="flash_attention_2"
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # SAM2 initialization
            SAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if SAM2_DIR not in sys.path:
                sys.path.append(SAM2_DIR)

            sam2_checkpoint = os.path.abspath(os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_base_plus.pt"))
            model_cfg = os.path.abspath(os.path.join(SAM2_DIR, "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"))

            config_dir = os.path.abspath(os.path.dirname(model_cfg))
            config_name = os.path.basename(model_cfg)

            GlobalHydra.instance().clear()

            with initialize_config_dir(config_dir=config_dir):
                self.predictor = build_sam2_camera_predictor(
                    config_name,
                    sam2_checkpoint,
                    device=self.device
                )

            self.ann_obj_id = 1
            self.initialized = True

        except Exception as e:
            print(f"Error in initialize_models: {str(e)}")
            self.cleanup()
            raise

    def reinitialize(self):
        """Reinitialize the models after cleanup"""
        self.cleanup()
        self.initialize_models()

    def get_normalized_boxes(self, response_text: str) -> List[List[float]]:
        """Extract normalized coordinates from model response"""
        pattern = r'\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]'
        matches = re.findall(pattern, response_text)
        
        boxes = []
        for match in matches:
            try:
                box = [float(coord) for coord in match]
                if all(0 <= coord <= 1.0 for coord in box):
                    boxes.append(box)
            except ValueError:
                continue
        
        return boxes

    def get_pixel_boxes(self, normalized_boxes: List[List[float]], image_shape: Tuple[int, ...]) -> List[List[int]]:
        """Convert normalized coordinates to pixel coordinates"""
        height, width = image_shape[:2]
        pixel_boxes = []
        
        for box in normalized_boxes:
            x_min, y_min, x_max, y_max = box
            pixel_box = [
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height)
            ]
            pixel_boxes.append(pixel_box)
        
        return pixel_boxes

    def get_model_response(self, image: np.ndarray, prompt: str) -> str:
        """Get response from Qwen2-VL model with memory management"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            system_prompt = """You are a helpful assistant to detect objects in images. Please follow these rules:
1. Return coordinates normalized between 0 and 1
2. Format: [[x_min, y_min, x_max, y_max], ...]
3. Be precise and accurate with coordinates
"""
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": system_prompt},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():  # Prevent memory leaks from gradients
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )

            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            # Clean up intermediate tensors
            del inputs
            del generated_ids
            del generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            print(f"Error in get_model_response: {str(e)}")
            raise

    def process_single_box(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """Generate mask for a single box"""
        try:
            box_array = np.array(box).reshape(1, 2, 2)
            
            _, _, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.ann_obj_id,
                bbox=box_array
            )
            
            if out_mask_logits is not None and len(out_mask_logits) > 0:
                # Convert to CPU immediately
                mask = out_mask_logits[0].cpu().numpy() if isinstance(out_mask_logits[0], torch.Tensor) else out_mask_logits[0]
                return mask > 0.0
            return None
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            return None

    def process_image_text(self, image: np.ndarray, prompt: str) -> List[np.ndarray]:
        """Process image and text with improved memory management"""
        try:
            if not self.initialized:
                self.initialize_models()
            
            print(f"Processing image with shape: {image.shape}")
            
            # Get model response and immediately clear Qwen model memory
            response = self.get_model_response(image, prompt)
            print(f"Model response: {response}")
            
            # Clear some memory after getting response
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process boxes
            normalized_boxes = self.get_normalized_boxes(response)
            if not normalized_boxes:
                print("No valid boxes found in response")
                return []
            
            pixel_boxes = self.get_pixel_boxes(normalized_boxes, image.shape)
            print(f"Found {len(pixel_boxes)} valid boxes")
            
            # Initialize SAM2
            self.predictor.load_first_frame(image)
            
            # Generate masks
            masks = []
            for box in pixel_boxes:
                mask = self.process_single_box(image, box)
                if mask is not None:
                    masks.append(mask)
                
                # Clear CUDA cache after each mask generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"Successfully generated {len(masks)} masks")
            
            # Reinitialize if memory usage is still high
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.5 * torch.cuda.get_device_properties(0).total_memory:
                self.reinitialize()
            
            return masks
            
        except Exception as e:
            print(f"Error in process_image_text: {str(e)}")
            self.reinitialize()
            return []

    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple masks into one with error handling"""
        try:
            if not masks:
                return None
            
            # Ensure all masks have the same shape
            shape = masks[0].shape
            valid_masks = [mask for mask in masks if mask.shape == shape]
            
            if not valid_masks:
                return None
            
            combined = np.zeros_like(valid_masks[0], dtype=bool)
            for mask in valid_masks:
                combined = combined | mask
                
            return combined
            
        except Exception as e:
            print(f"Error in combine_masks: {str(e)}")
            return None

    def track(self, image: np.ndarray) -> tuple:
        """Track objects in image with memory management"""
        try:
            if not self.initialized:
                self.initialize_models()
                
            self.predictor.add_conditioning_frame(image)
            obj_ids, masks = self.predictor.track(image)
            
            # Move masks to CPU if they're on GPU
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            
            # Clean up if memory usage is high
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.99 * torch.cuda.get_device_properties(0).total_memory:
                self.reinitialize()
                
            return obj_ids, masks
            
        except Exception as e:
            print(f"Error in tracking: {str(e)}")
            self.reinitialize()
            return [], []
        
class LLaVA_Next_SAM2_Connector:
    def __init__(self):
        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        # LLaVA-NeXT 모델 로드
        self.model_name = "LLAVA/LLaVA-NeXT"
        
        # 올바른 모델 클래스와 프로세서 import
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True
        ).to(self.device)
        self.model.eval()

        # SAM2 초기화
        SAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if SAM2_DIR not in sys.path:
            sys.path.append(SAM2_DIR)

        sam2_checkpoint = os.path.abspath(os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_base_plus.pt"))
        model_cfg = os.path.abspath(os.path.join(SAM2_DIR, "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"))

        config_dir = os.path.abspath(os.path.dirname(model_cfg))
        config_name = os.path.basename(model_cfg)

        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir):
            self.predictor = build_sam2_camera_predictor(
                config_name,
                sam2_checkpoint,
                device=self.device
            )

        self.ann_obj_id = 1

    def get_normalized_boxes(self, response_text: str) -> List[List[float]]:
        """모델 응답에서 정규화된 좌표를 추출"""
        pattern = r'\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]'
        matches = re.findall(pattern, response_text)
        
        boxes = []
        for match in matches:
            try:
                box = [float(coord) for coord in match]
                if all(0 <= coord <= 1.0 for coord in box):
                    boxes.append(box)
            except ValueError:
                continue
        
        return boxes

    def get_pixel_boxes(self, normalized_boxes: List[List[float]], image_shape: Tuple[int, ...]) -> List[List[int]]:
        """정규화된 좌표를 픽셀 좌표로 변환"""
        height, width = image_shape[:2]
        pixel_boxes = []
        
        for box in normalized_boxes:
            x_min, y_min, x_max, y_max = box
            pixel_box = [
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height)
            ]
            pixel_boxes.append(pixel_box)
        
        return pixel_boxes

    def get_model_response(self, image: np.ndarray, prompt: str) -> str:
        """LLaVA-NeXT 모델에서 응답 얻기"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 새로운 프롬프트 템플릿 사용
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Please detect objects in this image and return their coordinates in normalized format between 0 and 1. Format: [[x_min, y_min, x_max, y_max], ...]. Be precise and accurate with coordinates. " + prompt
                    },
                    {"type": "image"},
                ],
            }
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )

        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def process_single_box(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """단일 박스에 대한 마스크 생성"""
        try:
            # 박스를 SAM2가 원하는 형태로 변환 (1, 2, 2)
            box_array = np.array(box).reshape(1, 2, 2)
            
            _, _, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.ann_obj_id,
                bbox=box_array
            )
            
            if out_mask_logits is not None and len(out_mask_logits) > 0:
                return (out_mask_logits[0] > 0.0).cpu().numpy()
            return None
        except Exception as e:
            print(f"Error processing box {box}: {e}")
            return None

    def process_image_text(self, image: np.ndarray, prompt: str) -> List[np.ndarray]:
        """이미지와 텍스트 처리하여 마스크 생성"""
        try:
            print(f"Processing image with shape: {image.shape}")
            
            # 모델 응답 얻기
            response = self.get_model_response(image, prompt)
            print(f"Model response: {response}")
            
            # 정규화된 좌표 추출
            normalized_boxes = self.get_normalized_boxes(response)
            if not normalized_boxes:
                print("No valid boxes found in response")
                return []
            
            # 픽셀 좌표로 변환
            pixel_boxes = self.get_pixel_boxes(normalized_boxes, image.shape)
            print(f"Found {len(pixel_boxes)} valid boxes")
            
            # SAM2 초기화
            self.predictor.load_first_frame(image)
            
            # 각 박스별로 독립적으로 마스크 생성
            masks = []
            for box in pixel_boxes:
                mask = self.process_single_box(image, box)
                if mask is not None:
                    masks.append(mask)
            
            print(f"Successfully generated {len(masks)} masks")
            return masks
            
        except Exception as e:
            print(f"Error in process_image_text: {str(e)}")
            return []

    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """여러 마스크를 하나로 결합"""
        if not masks:
            return None
        
        combined = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined = combined | mask
        return combined

    def track(self, image: np.ndarray) -> tuple:
        """이미지에서 객체 추적"""
        try:
            self.predictor.add_conditioning_frame(image)
            obj_ids, masks = self.predictor.track(image)
            return obj_ids, masks
        except Exception as e:
            print(f"Error in tracking: {str(e)}")
            return [], []
