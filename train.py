'''
failed do not use it
'''
# import os
# import json
# import torch
# import logging
# import wandb
# from datetime import datetime
# from pathlib import Path
# from torch.utils.data import Dataset, DataLoader
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from peft import LoraConfig, get_peft_model
# from accelerate import Accelerator
# from tqdm import tqdm
# import numpy as np
# from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclass, field 
# import psutil
# from PIL import Image  # PIL에서 Image 모듈 임포트
# import re
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import get_linear_schedule_with_warmup  # scheduler 사용 시 필요
# from typing import Any  # Any 타입 사용 시 필요

# @dataclass
# class TrainingConfig:
#     # 기본 모델 설정
#     model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
#     vision_tower_name: str = "openai/clip-vit-large-patch14"
    
#     # LORA 관련 설정
#     lora_r: int = 8
#     lora_alpha: int = 16
#     lora_dropout: float = 0.1
#     lora_target_modules: List[str] = field(
#         default_factory=lambda: ["q_proj", "v_proj"]
#     )
    
#     # 학습 하이퍼파라미터
#     learning_rate: float = 1e-4
#     num_epochs: int = 10
#     train_batch_size: int = 4
#     eval_batch_size: int = 4
#     gradient_accumulation_steps: int = 2
#     warmup_ratio: float = 0.1  # warmup_steps 대신 warmup_ratio 사용
#     max_grad_norm: float = 1.0
#     save_steps: int = 500
#     eval_steps: int = 100
#     logging_steps: int = 10  # 추가된 설정
    
#     # 경로 설정
#     output_dir: str = './outputs'
#     checkpoint_dir: str = './checkpoints'
#     log_dir: str = './logs'
#     train_data_path: str = '/home/elicer/train_data.json'
#     val_data_path: str = '/home/elicer/val_data.json'
#     image_base_dir: str = '/home/elicer/person_train_images'
    
#     # Loss 관련 설정
#     coordinate_loss_weight: float = 1.0
#     positive_weight: float = 1.0
#     negative_weight: float = 0.8
#     sigma: float = 0.1
    
#     # 기타 설정
#     use_wandb: bool = True
#     seed: int = 42
#     fp16: bool = False
#     bf16: bool = True
#     num_workers: int = 4
#     use_flash_attention: bool = True
#     max_length: int = 2048
#     min_valid_points: int = 1
    
#     # 이미지 크기 설정 추가
#     image_size: Tuple[int, int] = (224, 224)  # 기본값 설정
    
#     def __post_init__(self):
#         """설정값 검증"""
#         if self.gradient_accumulation_steps < 1:
#             raise ValueError("gradient_accumulation_steps must be >= 1")
#         if self.train_batch_size < 1:
#             raise ValueError("train_batch_size must be >= 1")
#         if self.eval_batch_size < 1:
#             raise ValueError("eval_batch_size must be >= 1")
#         # image_size 검증 추가
#         if not isinstance(self.image_size, tuple) or len(self.image_size) != 2:
#             raise ValueError("image_size must be a tuple of two integers (width, height)")

#     # 기본 모델 설정
#     model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
#     vision_tower_name: str = "openai/clip-vit-large-patch14"
    
#     # LORA 관련 설정
#     lora_r: int = 8
#     lora_alpha: int = 16
#     lora_dropout: float = 0.1
#     lora_target_modules: List[str] = field(
#         default_factory=lambda: ["q_proj", "v_proj"]
#     )
    
#     # 학습 하이퍼파라미터
#     learning_rate: float = 1e-4
#     num_epochs: int = 10
#     train_batch_size: int = 4
#     eval_batch_size: int = 4
#     gradient_accumulation_steps: int = 2
#     warmup_ratio: float = 0.1  # warmup_steps 대신 warmup_ratio 사용
#     max_grad_norm: float = 1.0
#     save_steps: int = 500
#     eval_steps: int = 100
#     logging_steps: int = 10  # 추가된 설정
    
#     # 경로 설정
#     output_dir: str = './outputs'
#     checkpoint_dir: str = './checkpoints'
#     log_dir: str = './logs'
#     train_data_path: str = '/home/elicer/train_data.json'
#     val_data_path: str = '/home/elicer/val_data.json'
#     image_base_dir: str = '/home/elicer/person_train_images'
    
#     # Loss 관련 설정
#     coordinate_loss_weight: float = 1.0
#     positive_weight: float = 1.0
#     negative_weight: float = 0.8
#     sigma: float = 0.1
    
#     # 기타 설정
#     use_wandb: bool = True
#     seed: int = 42
#     fp16: bool = False
#     bf16: bool = True
#     num_workers: int = 4
#     use_flash_attention: bool = True
#     max_length: int = 2048
#     min_valid_points: int = 1
    
#     # 이미지 크기 설정 추가
#     image_size: Tuple[int, int] = (224, 224)  # 기본값 설정
    
#     def __post_init__(self):
#         """설정값 검증"""
#         if self.gradient_accumulation_steps < 1:
#             raise ValueError("gradient_accumulation_steps must be >= 1")
#         if self.train_batch_size < 1:
#             raise ValueError("train_batch_size must be >= 1")
#         if self.eval_batch_size < 1:
#             raise ValueError("eval_batch_size must be >= 1")


# class SegmentationDataset(Dataset):
#     def __init__(
#         self, 
#         json_path: str,
#         processor: AutoProcessor,
#         image_base_dir: str,
#         image_size: Tuple[int, int] = (224, 224),
#         min_valid_points: int = 1
#     ):
#         """
#         Args:
#             json_path: JSON 파일 경로
#             processor: Qwen 프로세서 (이미 special tokens가 추가된 상태)
#             image_base_dir: 이미지 파일들의 기본 디렉토리
#             image_size: 목표 이미지 크기 (width, height)
#             min_valid_points: 최소 유효 포인트 수
#         """
#         # 경로 검증
#         json_path = Path(json_path)
#         if not json_path.exists():
#             raise FileNotFoundError(f"Data file not found: {json_path}")
            
#         self.image_base_dir = Path(image_base_dir)
#         if not self.image_base_dir.exists():
#             raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
            
#         # 데이터 로드
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 self.data = json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON file: {str(e)}")
            
#         self.processor = processor
#         self.tokenizer = processor.tokenizer
#         self.image_size = image_size
#         self.min_valid_points = min_valid_points
        
#         # 유효한 데이터만 필터링
#         self.filter_valid_data()
        
#         # 데이터셋 분석
#         self.analyze_dataset()
        
#     def filter_valid_data(self):
#         """유효한 데이터만 필터링"""
#         valid_data = []
#         original_count = len(self.data)
        
#         for idx, item in enumerate(self.data):
#             try:
#                 # 1. 메시지 추출 검증
#                 user_messages = [msg for msg in item['messages'] if msg['role'] == 'user']
#                 assistant_messages = [msg for msg in item['messages'] if msg['role'] == 'assistant']
                
#                 if not user_messages or not assistant_messages:
#                     logging.warning(f"Sample {idx} missing user or assistant messages.")
#                     continue
                
#                 user_message = user_messages[0]
#                 assistant_message = assistant_messages[0]
                
#                 # 2. 이미지 경로 검증
#                 content = user_message['content']
#                 if not isinstance(content, list):
#                     logging.warning(f"Sample {idx} user content is not a list.")
#                     continue
                
#                 image_rel_path = next((c['image'] for c in content if c['type'] == 'image'), None)
#                 if not image_rel_path:
#                     logging.warning(f"Sample {idx} missing image path.")
#                     continue
                    
#                 image_path = self.image_base_dir / image_rel_path
#                 if not image_path.exists():
#                     logging.warning(f"Sample {idx} image not found: {image_rel_path}")
#                     continue
                
#                 # 3. 포인트 데이터 검증
#                 segmentation_points = next((c for c in assistant_message['content'] if c['type'] == 'segmentation_points'), None)
#                 if not segmentation_points:
#                     logging.warning(f"Sample {idx} missing segmentation points.")
#                     continue
                
#                 num_positive = len(segmentation_points['points']['positive'])
#                 num_negative = len(segmentation_points['points']['negative'])
                
#                 if num_positive < self.min_valid_points:
#                     logging.warning(f"Sample {idx} not enough positive points: {num_positive}")
#                     continue
                    
#                 if num_negative < self.min_valid_points:
#                     logging.warning(f"Sample {idx} not enough negative points: {num_negative}")
#                     continue
                
#                 # 4. 추가적인 유효성 검사 (예: 텍스트 필드 존재 여부)
#                 response_text = next((c['text'] for c in assistant_message['content'] if c['type'] == 'text'), None)
#                 prompt = next((c['text'] for c in user_message['content'] if c['type'] == 'text'), None)
#                 if not response_text or not prompt:
#                     logging.warning(f"Sample {idx} missing response or prompt text.")
#                     continue
                
#                 # 모든 검증을 통과한 데이터만 추가
#                 valid_data.append(item)
                
#             except (KeyError, IndexError, Exception) as e:
#                 logging.error(f"Error processing item {idx}: {str(e)}")
#                 continue
        
#         logging.info(f"\nData Filtering Results:")
#         logging.info(f"Original samples: {original_count}")
#         logging.info(f"Valid samples: {len(valid_data)}")
#         logging.info(f"Filtered out: {original_count - len(valid_data)} samples\n")
        
#         self.data = valid_data
        
#     def analyze_dataset(self):
#         """데이터셋 통계 분석"""
#         self.max_positive_points = 0
#         self.max_negative_points = 0
#         valid_samples = 0
        
#         for idx, item in enumerate(self.data):
#             try:
#                 assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
#                 points_data = next((c for c in assistant_message['content'] if c['type'] == 'segmentation_points'), None)
                
#                 if not points_data:
#                     continue
                
#                 num_positive = len(points_data['points']['positive'])
#                 num_negative = len(points_data['points']['negative'])
                
#                 if num_positive >= self.min_valid_points and num_negative >= self.min_valid_points:
#                     valid_samples += 1
                    
#                 self.max_positive_points = max(self.max_positive_points, num_positive)
#                 self.max_negative_points = max(self.max_negative_points, num_negative)
                
#             except (KeyError, IndexError) as e:
#                 logging.error(f"Error analyzing item {idx}: {str(e)}")
#                 continue
                
#         logging.info(f"Dataset Statistics:")
#         logging.info(f"Total samples: {len(self.data)}")
#         logging.info(f"Valid samples: {valid_samples}")
#         logging.info(f"Maximum positive points: {self.max_positive_points}")
#         logging.info(f"Maximum negative points: {self.max_negative_points}")
    
#     def normalize_coordinates(
#         self,
#         points: List[List[float]],
#         original_size: Tuple[int, int]
#     ) -> List[List[float]]:
#         """좌표를 0~1 사이의 비율로 정규화"""
#         orig_width, orig_height = original_size
        
#         normalized_points = []
#         for x, y in points:
#             # 원본 이미지에서의 비율로 변환 및 클리핑
#             norm_x = max(0.0, min(1.0, x / orig_width))
#             norm_y = max(0.0, min(1.0, y / orig_height))
#             normalized_points.append([norm_x, norm_y])
        
#         return normalized_points
    
#     def format_coordinates(
#         self,
#         points_data: Dict,
#         original_size: Tuple[int, int]
#     ) -> Tuple[str, List[List[float]], List[List[float]]]:
#         """정규화된 좌표를 구조화된 텍스트로 변환"""
#         positive_points = points_data['points']['positive']
#         negative_points = points_data['points']['negative']
        
#         # 좌표 정규화
#         norm_positive = self.normalize_coordinates(positive_points, original_size)
#         norm_negative = self.normalize_coordinates(negative_points, original_size)
        
#         # 텍스트 포맷팅
#         coord_text = "<coord_start>"
#         for x, y in norm_positive:
#             coord_text += f"<pos><x>{x:.4f}<y>{y:.4f}>"
#         for x, y in norm_negative:
#             coord_text += f"<neg><x>{x:.4f}<y>{y:.4f}>"
#         coord_text += "<coord_end>"
        
#         return coord_text, norm_positive, norm_negative
    
#     def validate_points(
#         self,
#         positive_points: List[List[float]],
#         negative_points: List[List[float]]
#     ) -> bool:
#         """포인트 데이터 유효성 검사"""
#         if len(positive_points) < self.min_valid_points:
#             return False
#         if len(negative_points) < self.min_valid_points:
#             return False
            
#         # 좌표값 검사
#         for points in [positive_points, negative_points]:
#             for x, y in points:
#                 if not (0 <= x <= 1 and 0 <= y <= 1):
#                     return False
#         return True
    
#     def process_image(self, image_path: Path) -> Optional[Image.Image]:
#         """이미지 로드 및 전처리"""
#         try:
#             image = Image.open(image_path).convert('RGB')
#             image = image.resize(self.image_size, Image.BICUBIC)
#             return image
                
#         except Exception as e:
#             logging.error(f"Error processing image {image_path}: {str(e)}")
#             return None
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         item = self.data[idx]
        
#         try:
#             # 1. 메시지 추출
#             user_message = [msg for msg in item['messages'] if msg['role'] == 'user'][0]
#             assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
            
#             # 2. 이미지 경로 및 프롬프트 처리
#             content = user_message['content']
#             image_rel_path = next((c['image'] for c in content if c['type'] == 'image'), None)
#             prompt = next((c['text'] for c in content if c['type'] == 'text'), None)
            
#             if not image_rel_path or not prompt:
#                 logging.error(f"Sample {idx} missing image path or prompt.")
#                 raise KeyError(f"Sample {idx} missing image path or prompt.")
            
#             # 3. 이미지 처리
#             image_path = self.image_base_dir / image_rel_path
#             image = self.process_image(image_path)
#             if image is None:
#                 logging.error(f"Sample {idx} failed to process image.")
#                 raise ValueError(f"Sample {idx} failed to process image.")
#             original_size = image.size
            
#             # 4. 포인트 데이터 처리
#             content = assistant_message['content']
#             response_text = next((c['text'] for c in content if c['type'] == 'text'), None)
#             points_data = next((c for c in content if c['type'] == 'segmentation_points'), None)
            
#             if not response_text or not points_data:
#                 logging.error(f"Sample {idx} missing response text or segmentation points.")
#                 raise KeyError(f"Sample {idx} missing response text or segmentation points.")
            
#             # 5. 좌표 정규화 및 텍스트 변환
#             coord_text, norm_positive, norm_negative = self.format_coordinates(points_data, original_size)
            
#             # 6. 좌표 유효성 검사 (이미 필터링됨)
#             # (필요 없음)
            
#             # 7. 전체 대화 구성
#             conversation = f"User: {prompt}\nAssistant: {response_text} {coord_text}"
            
#             # 8. 입력 처리
#             inputs = self.processor(
#                 text=conversation,
#                 images=image,
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.processor.tokenizer.model_max_length  # 모델의 max_length에 맞게 설정
#             )
            
#             # FeatureExtractionUtils 또는 BatchFeature 객체를 딕셔너리로 변환
#             if hasattr(inputs, 'to_dict'):
#                 inputs = inputs.to_dict()
#             elif isinstance(inputs, dict):
#                 pass  # 이미 딕셔너리인 경우
#             else:
#                 inputs = {k: v for k, v in inputs.items()}
            
#             # 9. 레이블 생성
#             input_ids = inputs['input_ids'].squeeze(0)
#             labels = input_ids.clone()
            
#             # 10. Assistant 토큰 위치 찾기
#             assistant_token_id = self.tokenizer.encode("Assistant:", add_special_tokens=False)
#             assistant_start_idx = None
#             for i in range(len(input_ids) - len(assistant_token_id) + 1):
#                 if input_ids[i:i+len(assistant_token_id)].tolist() == assistant_token_id:
#                     assistant_start_idx = i + len(assistant_token_id)
#                     break
            
#             if assistant_start_idx is None:
#                 logging.error(f"Assistant token not found in the input for sample {idx}.")
#                 raise ValueError(f"Assistant token not found in the input for sample {idx}.")
            
#             # 11. 레이블 마스킹
#             labels[:assistant_start_idx] = -100
            
#             # 12. 레이블 추가
#             inputs['labels'] = labels
#             logging.debug(f"Labels added for index {idx}")
            
#             # 13. 배치 차원 제거
#             for k, v in inputs.items():
#                 if torch.is_tensor(v) and v.ndim > 0:
#                     inputs[k] = v.squeeze(0)
            
#             # 14. 좌표 및 마스크 추가
#             inputs['positive_points'] = torch.tensor(norm_positive, dtype=torch.float)
#             inputs['negative_points'] = torch.tensor(norm_negative, dtype=torch.float)
            
#             pos_mask = torch.ones(len(norm_positive), dtype=torch.float)
#             neg_mask = torch.ones(len(norm_negative), dtype=torch.float)
            
#             inputs['positive_mask'] = pos_mask
#             inputs['negative_mask'] = neg_mask
            
#             # 15. 크기 정보 추가
#             inputs['original_size'] = torch.tensor(original_size, dtype=torch.float)
#             inputs['target_size'] = torch.tensor(self.image_size, dtype=torch.float)
            
#             return inputs
#     def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         """배치 데이터 처리 개선"""
#         try:
#             # 모든 샘플에 'labels' 키가 존재하는지 확인
#             for idx, item in enumerate(batch):
#                 if 'labels' not in item:
#                     logging.error(f"Sample at index {idx} is missing 'labels' key.")
#                     raise KeyError(f"Sample at index {idx} is missing 'labels' key.")

#             # 1. 시퀀스 데이터 패딩
#             input_ids = torch.nn.utils.rnn.pad_sequence(
#                 [item['input_ids'] for item in batch], 
#                 batch_first=True, 
#                 padding_value=0
#             )
#             attention_mask = torch.nn.utils.rnn.pad_sequence(
#                 [item['attention_mask'] for item in batch],
#                 batch_first=True,
#                 padding_value=0
#             )
#             labels = torch.nn.utils.rnn.pad_sequence(
#                 [item['labels'] for item in batch],
#                 batch_first=True,
#                 padding_value=-100
#             )
            
#             # 2. 이미지 데이터 스택
#             pixel_values = torch.stack([item['pixel_values'] for item in batch])
            
#             # 3. 좌표 데이터 패딩
#             max_pos = max(item['positive_points'].size(0) for item in batch)
#             max_neg = max(item['negative_points'].size(0) for item in batch)
            
#             def pad_points(points_list, max_len):
#                 padded = []
#                 masks = []
#                 for points in points_list:
#                     pad_size = max_len - points.size(0)
#                     if pad_size > 0:
#                         padding = torch.zeros(pad_size, 2, device=points.device)
#                         points_padded = torch.cat([points, padding], dim=0)
#                         mask = torch.cat([torch.ones(points.size(0)), torch.zeros(pad_size)])
#                     else:
#                         points_padded = points
#                         mask = torch.ones(points.size(0))
#                     padded.append(points_padded)
#                     masks.append(mask)
#                 return torch.stack(padded), torch.stack(masks)
            
#             positive_points, positive_masks = pad_points(
#                 [item['positive_points'] for item in batch], max_pos
#             )
#             negative_points, negative_masks = pad_points(
#                 [item['negative_points'] for item in batch], max_neg
#             )
            
#             # 4. 크기 정보 스택
#             original_sizes = torch.stack([item['original_size'] for item in batch])
#             target_sizes = torch.stack([item['target_size'] for item in batch])
            
#             return {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'labels': labels,
#                 'pixel_values': pixel_values,
#                 'positive_points': positive_points,
#                 'negative_points': negative_points,
#                 'positive_mask': positive_masks,
#                 'negative_mask': negative_masks,
#                 'original_size': original_sizes,
#                 'target_size': target_sizes
#             }
            
#         except KeyError as e:
#             logging.error(f"Error in collate_fn: {str(e)}")
#             raise e
#         except Exception as e:
#             logging.error(f"Unexpected error in collate_fn: {str(e)}")
#             raise e

#     def __init__(
#         self, 
#         json_path: str,
#         processor: AutoProcessor,
#         image_base_dir: str,
#         image_size: Tuple[int, int] = (224, 224),
#         min_valid_points: int = 1
#     ):
#         """
#         Args:
#             json_path: JSON 파일 경로
#             processor: Qwen 프로세서 (이미 special tokens가 추가된 상태)
#             image_base_dir: 이미지 파일들의 기본 디렉토리
#             image_size: 목표 이미지 크기 (width, height)
#             min_valid_points: 최소 유효 포인트 수
#         """
#         # 경로 검증
#         json_path = Path(json_path)
#         if not json_path.exists():
#             raise FileNotFoundError(f"Data file not found: {json_path}")
            
#         self.image_base_dir = Path(image_base_dir)
#         if not self.image_base_dir.exists():
#             raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
            
#         # 데이터 로드
#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 self.data = json.load(f)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON file: {str(e)}")
            
#         self.processor = processor
#         self.tokenizer = processor.tokenizer
#         self.image_size = image_size
#         self.min_valid_points = min_valid_points
        
#         # 유효한 데이터만 필터링
#         self.filter_valid_data()
        
#         # 데이터셋 분석
#         self.analyze_dataset()
        
#     def filter_valid_data(self):
#         """유효한 데이터만 필터링"""
#         valid_data = []
#         original_count = len(self.data)
        
#         for idx, item in enumerate(self.data):
#             try:
#                 # 1. 메시지 추출 검증
#                 user_message = [msg for msg in item['messages'] if msg['role'] == 'user'][0]
#                 assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
                
#                 # 2. 이미지 경로 검증
#                 content = user_message['content']
#                 if not isinstance(content, list):
#                     continue
                
#                 image_rel_path = next((c['image'] for c in content if c['type'] == 'image'), None)
#                 if not image_rel_path:
#                     continue
                    
#                 image_path = self.image_base_dir / image_rel_path
#                 if not image_path.exists():
#                     print(f"Image not found: {image_rel_path}")
#                     continue
                
#                 # 3. 포인트 데이터 검증
#                 points_data = [c for c in assistant_message['content'] if c['type'] == 'segmentation_points'][0]
#                 num_positive = len(points_data['points']['positive'])
#                 num_negative = len(points_data['points']['negative'])
                
#                 if num_positive < self.min_valid_points:
#                     print(f"Not enough positive points: {num_positive}")
#                     continue
                    
#                 if num_negative < self.min_valid_points:
#                     print(f"Not enough negative points: {num_negative}")
#                     continue
                
#                 # 4. 추가적인 유효성 검사 (예: 텍스트 필드 존재 여부)
#                 response_text = next((c['text'] for c in assistant_message['content'] if c['type'] == 'text'), None)
#                 prompt = next((c['text'] for c in user_message['content'] if c['type'] == 'text'), None)
#                 if not response_text or not prompt:
#                     continue
                
#                 # 모든 검증을 통과한 데이터만 추가
#                 valid_data.append(item)
                
#             except (KeyError, IndexError, Exception) as e:
#                 print(f"Error processing item {idx}: {str(e)}")
#                 continue
        
#         print(f"\nData Filtering Results:")
#         print(f"Original samples: {original_count}")
#         print(f"Valid samples: {len(valid_data)}")
#         print(f"Filtered out: {original_count - len(valid_data)} samples\n")
        
#         self.data = valid_data
        
#     def analyze_dataset(self):
#         """데이터셋 통계 분석"""
#         self.max_positive_points = 0
#         self.max_negative_points = 0
#         valid_samples = 0
        
#         for item in self.data:
#             try:
#                 assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
#                 points_data = [c for c in assistant_message['content'] if c['type'] == 'segmentation_points'][0]
                
#                 num_positive = len(points_data['points']['positive'])
#                 num_negative = len(points_data['points']['negative'])
                
#                 if num_positive >= self.min_valid_points and num_negative >= self.min_valid_points:
#                     valid_samples += 1
                    
#                 self.max_positive_points = max(self.max_positive_points, num_positive)
#                 self.max_negative_points = max(self.max_negative_points, num_negative)
                
#             except (KeyError, IndexError):
#                 continue
                
#         print(f"Dataset Statistics:")
#         print(f"Total samples: {len(self.data)}")
#         print(f"Valid samples: {valid_samples}")
#         print(f"Maximum positive points: {self.max_positive_points}")
#         print(f"Maximum negative points: {self.max_negative_points}")
    
#     def normalize_coordinates(
#         self,
#         points: List[List[float]],
#         original_size: Tuple[int, int]
#     ) -> List[List[float]]:
#         """좌표를 0~1 사이의 비율로 정규화"""
#         orig_width, orig_height = original_size
        
#         normalized_points = []
#         for x, y in points:
#             # 원본 이미지에서의 비율로 변환 및 클리핑
#             norm_x = max(0.0, min(1.0, x / orig_width))
#             norm_y = max(0.0, min(1.0, y / orig_height))
#             normalized_points.append([norm_x, norm_y])
        
#         return normalized_points
    
#     def format_coordinates(
#         self,
#         points_data: Dict,
#         original_size: Tuple[int, int]
#     ) -> Tuple[str, List[List[float]], List[List[float]]]:
#         """정규화된 좌표를 구조화된 텍스트로 변환"""
#         positive_points = points_data['points']['positive']
#         negative_points = points_data['points']['negative']
        
#         # 좌표 정규화
#         norm_positive = self.normalize_coordinates(positive_points, original_size)
#         norm_negative = self.normalize_coordinates(negative_points, original_size)
        
#         # 텍스트 포맷팅
#         coord_text = "<coord_start>"
#         for x, y in norm_positive:
#             coord_text += f"<pos><x>{x:.4f}<y>{y:.4f}"
#         for x, y in norm_negative:
#             coord_text += f"<neg><x>{x:.4f}<y>{y:.4f}"
#         coord_text += "<coord_end>"
        
#         return coord_text, norm_positive, norm_negative
    
#     def validate_points(
#         self,
#         positive_points: List[List[float]],
#         negative_points: List[List[float]]
#     ) -> bool:
#         """포인트 데이터 유효성 검사"""
#         if len(positive_points) < self.min_valid_points:
#             return False
#         if len(negative_points) < self.min_valid_points:
#             return False
            
#         # 좌표값 검사
#         for points in [positive_points, negative_points]:
#             for x, y in points:
#                 if not (0 <= x <= 1 and 0 <= y <= 1):
#                     return False
#         return True
    
#     def process_image(self, image_path: Path) -> Optional[Image.Image]:
#         """이미지 로드 및 전처리"""
#         try:
#             image = Image.open(image_path).convert('RGB')
#             image = image.resize(self.image_size, Image.BICUBIC)
#             return image
            
#         except Exception as e:
#             print(f"Error processing image {image_path}: {str(e)}")
#             return None
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         item = self.data[idx]
        
#         # 1. 메시지 추출
#         user_message = [msg for msg in item['messages'] if msg['role'] == 'user'][0]
#         assistant_message = [msg for msg in item['messages'] if msg['role'] == 'assistant'][0]
        
#         # 2. 이미지 경로 및 프롬프트 처리
#         content = user_message['content']
#         image_rel_path = next((c['image'] for c in content if c['type'] == 'image'), None)
#         prompt = next((c['text'] for c in content if c['type'] == 'text'), None)
        
#         # 3. 이미지 처리
#         image_path = self.image_base_dir / image_rel_path
#         image = self.process_image(image_path)
#         original_size = image.size
        
#         # 4. 포인트 데이터 처리
#         content = assistant_message['content']
#         response_text = next((c['text'] for c in content if c['type'] == 'text'), None)
#         points_data = next((c for c in content if c['type'] == 'segmentation_points'), None)
        
#         # 5. 좌표 정규화 및 텍스트 변환
#         coord_text, norm_positive, norm_negative = self.format_coordinates(points_data, original_size)
        
#         # 6. 좌표 유효성 검사 (이미 필터링됨)
        
#         # 7. 전체 대화 구성
#         conversation = f"User: {prompt}\nAssistant: {response_text} {coord_text}"
        
#         # 8. 입력 처리
#         inputs = self.processor(
#             text=conversation,
#             images=image,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=2048
#         )
        
#         # 9. 레이블 생성
#         input_ids = inputs['input_ids'].squeeze(0)
#         labels = input_ids.clone()
        
#         # 10. Assistant 토큰 위치 찾기
#         assistant_token_id = self.tokenizer.encode("Assistant:", add_special_tokens=False)
#         assistant_start_idx = None
#         for i in range(len(input_ids) - len(assistant_token_id) + 1):
#             if input_ids[i:i+len(assistant_token_id)].tolist() == assistant_token_id:
#                 assistant_start_idx = i + len(assistant_token_id)
#                 break
        
#         if assistant_start_idx is None:
#             raise ValueError("Assistant token not found in the input.")
        
#         # 11. 레이블 마스킹
#         labels[:assistant_start_idx] = -100
        
#         # 12. 배치 차원 제거
#         for k, v in inputs.items():
#             if torch.is_tensor(v) and v.ndim > 0:
#                 inputs[k] = v.squeeze(0)
        
#         # 13. 좌표 및 마스크 추가
#         inputs['positive_points'] = torch.tensor(norm_positive, dtype=torch.float)
#         inputs['negative_points'] = torch.tensor(norm_negative, dtype=torch.float)
        
#         pos_mask = torch.ones(len(norm_positive), dtype=torch.float)
#         neg_mask = torch.ones(len(norm_negative), dtype=torch.float)
        
#         inputs['positive_mask'] = pos_mask
#         inputs['negative_mask'] = neg_mask
        
#         # 14. 크기 정보 추가
#         inputs['original_size'] = torch.tensor(original_size, dtype=torch.float)
#         inputs['target_size'] = torch.tensor(self.image_size, dtype=torch.float)
        
#         return inputs
    
#     def __len__(self) -> int:
#         return len(self.data)

#     @staticmethod
#     def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         """배치 데이터 처리 개선"""
#         try:
#             # 1. 시퀀스 데이터 패딩
#             input_ids = torch.nn.utils.rnn.pad_sequence(
#                 [item['input_ids'] for item in batch], 
#                 batch_first=True, 
#                 padding_value=0
#             )
#             attention_mask = torch.nn.utils.rnn.pad_sequence(
#                 [item['attention_mask'] for item in batch],
#                 batch_first=True,
#                 padding_value=0
#             )
#             labels = torch.nn.utils.rnn.pad_sequence(
#                 [item['labels'] for item in batch],
#                 batch_first=True,
#                 padding_value=-100
#             )
            
#             # 2. 이미지 데이터 스택
#             pixel_values = torch.stack([item['pixel_values'] for item in batch])
            
#             # 3. 좌표 데이터 패딩
#             max_pos = max(item['positive_points'].size(0) for item in batch)
#             max_neg = max(item['negative_points'].size(0) for item in batch)
            
#             def pad_points(points_list, max_len):
#                 padded = []
#                 masks = []
#                 for points in points_list:
#                     pad_size = max_len - points.size(0)
#                     if pad_size > 0:
#                         padding = torch.zeros(pad_size, 2, device=points.device)
#                         points_padded = torch.cat([points, padding], dim=0)
#                         mask = torch.cat([torch.ones(points.size(0)), torch.zeros(pad_size)])
#                     else:
#                         points_padded = points
#                         mask = torch.ones(points.size(0))
#                     padded.append(points_padded)
#                     masks.append(mask)
#                 return torch.stack(padded), torch.stack(masks)
            
#             positive_points, positive_masks = pad_points(
#                 [item['positive_points'] for item in batch], max_pos
#             )
#             negative_points, negative_masks = pad_points(
#                 [item['negative_points'] for item in batch], max_neg
#             )
            
#             # 4. 크기 정보 스택
#             original_sizes = torch.stack([item['original_size'] for item in batch])
#             target_sizes = torch.stack([item['target_size'] for item in batch])
            
#             return {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask,
#                 'labels': labels,
#                 'pixel_values': pixel_values,
#                 'positive_points': positive_points,
#                 'negative_points': negative_points,
#                 'positive_mask': positive_masks,
#                 'negative_mask': negative_masks,
#                 'original_size': original_sizes,
#                 'target_size': target_sizes
#             }
            
#         except Exception as e:
#             logging.error(f"Error in collate_fn: {str(e)}")
#             raise e

# class SegmentationLoss(nn.Module):
#     def __init__(
#         self,
#         positive_weight: float = 1.0,
#         negative_weight: float = 1.0,
#         sigma: float = 0.1,
#         use_dynamic_weights: bool = True,
#         min_valid_points: int = 1
#     ):
#         super().__init__()
        
#         if use_dynamic_weights:
#             self.positive_weight = nn.Parameter(torch.tensor([positive_weight]))
#             self.negative_weight = nn.Parameter(torch.tensor([negative_weight]))
#             self.distance_weight = nn.Parameter(torch.tensor([0.1]))
#         else:
#             self.register_buffer('positive_weight', torch.tensor([positive_weight]))
#             self.register_buffer('negative_weight', torch.tensor([negative_weight]))
#             self.register_buffer('distance_weight', torch.tensor([0.1]))
            
#         self.sigma = sigma
#         self.min_valid_points = min_valid_points
#         self.eps = 1e-6
        
#     def forward(
#         self,
#         pred_coordinates: Dict[str, torch.Tensor],
#         target: Dict[str, torch.Tensor],
#         masks: Optional[Dict[str, torch.Tensor]] = None
#     ) -> Dict[str, torch.Tensor]:
#         try:
#             if pred_coordinates['positive_points'].size(1) == 0 or pred_coordinates['negative_points'].size(1) == 0:
#                 return self._get_zero_loss(pred_coordinates['positive_points'].device)
                
#             pos_mask = masks.get('positive_mask') if masks else None
#             neg_mask = masks.get('negative_mask') if masks else None
            
#             pos_loss = self.compute_positive_loss(
#                 pred_coordinates['positive_points'],
#                 target['positive_points'],
#                 pos_mask
#             )
            
#             neg_loss = self.compute_negative_loss(
#                 pred_coordinates['negative_points'],
#                 target['negative_points'],
#                 neg_mask
#             )
            
#             spread_loss = self.compute_spread_loss(
#                 pred_coordinates['positive_points'],
#                 pred_coordinates['negative_points'],
#                 pos_mask,
#                 neg_mask
#             )
            
#             if isinstance(self.positive_weight, nn.Parameter):
#                 total_loss = (
#                     torch.exp(-self.positive_weight) * pos_loss + self.positive_weight +
#                     torch.exp(-self.negative_weight) * neg_loss + self.negative_weight +
#                     torch.exp(-self.distance_weight) * spread_loss + self.distance_weight
#                 )
#             else:
#                 total_loss = (
#                     self.positive_weight * pos_loss +
#                     self.negative_weight * neg_loss +
#                     self.distance_weight * spread_loss
#                 )
            
#             return {
#                 'loss': total_loss,
#                 'pos_loss': pos_loss.item(),
#                 'neg_loss': neg_loss.item(),
#                 'spread_loss': spread_loss.item(),
#                 'pos_weight': (torch.exp(-self.positive_weight).item() 
#                              if isinstance(self.positive_weight, nn.Parameter) 
#                              else self.positive_weight.item()),
#                 'neg_weight': (torch.exp(-self.negative_weight).item() 
#                              if isinstance(self.negative_weight, nn.Parameter) 
#                              else self.negative_weight.item())
#             }
            
#         except Exception as e:
#             logging.error(f"Error in loss computation: {str(e)}")
#             return self._get_zero_loss(pred_coordinates['positive_points'].device)

#     def _get_zero_loss(self, device: torch.device) -> Dict[str, torch.Tensor]:
#         """기본 손실값 반환"""
#         return {
#             'loss': torch.tensor(0.0, device=device, requires_grad=True),
#             'pos_loss': 0.0,
#             'neg_loss': 0.0,
#             'spread_loss': 0.0,
#             'pos_weight': 1.0,
#             'neg_weight': 1.0
#         }
        
#     def compute_positive_loss(self, pred_pos: torch.Tensor, target_pos: torch.Tensor, pos_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         """Positive 좌표 손실 계산"""
#         if pos_mask is not None:
#             pred_pos = pred_pos * pos_mask.unsqueeze(-1)
#             target_pos = target_pos * pos_mask.unsqueeze(-1)
        
#         loss = F.mse_loss(pred_pos, target_pos, reduction='mean')
#         return loss
    
#     def compute_negative_loss(self, pred_neg: torch.Tensor, target_neg: torch.Tensor, neg_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         """Negative 좌표 손실 계산"""
#         if neg_mask is not None:
#             pred_neg = pred_neg * neg_mask.unsqueeze(-1)
#             target_neg = target_neg * neg_mask.unsqueeze(-1)
        
#         loss = F.mse_loss(pred_neg, target_neg, reduction='mean')
#         return loss
    
#     def compute_spread_loss(self, pred_pos: torch.Tensor, pred_neg: torch.Tensor, pos_mask: Optional[torch.Tensor], neg_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         """Positive와 Negative 좌표 간 거리 손실 계산"""
#         distances = self.compute_pairwise_distances(pred_pos, pred_neg)
#         loss = F.relu(self.sigma - distances).mean()
#         return loss
    
#     def compute_pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """배치 단위의 페어와이즈 거리 계산"""
#         if x.size(0) == 0 or y.size(0) == 0:
#             return torch.zeros(x.size(0), y.size(0), device=x.device)
            
#         x_norm = (x ** 2).sum(1).unsqueeze(1)
#         y_norm = (y ** 2).sum(1).unsqueeze(0)
        
#         dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.transpose(0, 1))
#         dist = torch.clamp(dist, min=0.0)
        
#         return torch.sqrt(dist + self.eps)

# class EnhancedLogger:
#     def __init__(self, config: TrainingConfig):
#         self.config = config
#         self.log_dir = Path(config.log_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         self.log_dir.mkdir(parents=True, exist_ok=True)
        
#         # 로그 파일 설정
#         self.setup_file_logging()
        
#         # wandb 설정
#         if config.use_wandb:
#             self.setup_wandb()
        
#         self.log("Logger initialized successfully")
#         self.log_system_info()
    
#     def setup_file_logging(self):
#         """파일 로깅 설정"""
#         self.logger = logging.getLogger('training')
#         self.logger.setLevel(logging.INFO)
        
#         # 파일 핸들러
#         fh = logging.FileHandler(self.log_dir / 'training.log')
#         fh.setLevel(logging.INFO)
        
#         # 콘솔 핸들러
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
        
#         # 포맷터
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         fh.setFormatter(formatter)
#         ch.setFormatter(formatter)
        
#         self.logger.addHandler(fh)
#         self.logger.addHandler(ch)
    
#     def setup_wandb(self):
#         """Weights & Biases 설정"""
#         wandb.init(
#             project=f"segmentation_{datetime.now().strftime('%Y%m%d')}",
#             config=self.config.__dict__,
#             name=f"run_{datetime.now().strftime('%H%M%S')}",
#         )
#         wandb.watch(self, log="all")
    
#     def log_system_info(self):
#         """시스템 정보 로깅"""
#         if torch.cuda.is_available():
#             gpu_info = {
#                 "gpu_name": torch.cuda.get_device_name(0),
#                 "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB",
#                 "cuda_version": torch.version.cuda,
#             }
#             self.log(f"GPU Info: {gpu_info}")
        
#         cpu_info = {
#             "cpu_count": psutil.cpu_count(),
#             "memory_total": f"{psutil.virtual_memory().total / 1e9:.2f}GB",
#         }
#         self.log(f"CPU Info: {cpu_info}")
    
#     def log(self, message: str, level: str = "info"):
#         """로그 메시지 기록"""
#         if level == "info":
#             self.logger.info(message)
#         elif level == "debug":
#             self.logger.debug(message)
#         elif level == "warning":
#             self.logger.warning(message)
#         elif level == "error":
#             self.logger.error(message)
#         else:
#             self.logger.info(message)
    
#     def log_metrics(self, metrics: Dict[str, float], step: int):
#         """메트릭 기록"""
#         # 로그 파일에 기록
#         metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
#         self.log(f"Step {step} - {metrics_str}")
        
#         # wandb에 기록
#         if self.config.use_wandb:
#             wandb.log(metrics, step=step)

# def train(config: TrainingConfig):
#     """전체 학습 프로세스"""
#     # Logger 초기화
#     logger = EnhancedLogger(config)
#     logger.log("Starting training preparation...")
    
#     try:
#         # Accelerator 초기화
#         logger.log("Initializing Accelerator...")
#         accelerator = Accelerator(
#             gradient_accumulation_steps=config.gradient_accumulation_steps,
#             log_with="wandb" if config.use_wandb else None
#         )
        
#         # Processor 및 Tokenizer 초기화
#         logger.log("Initializing processor and tokenizer...")
#         processor = AutoProcessor.from_pretrained(config.model_name)
#         tokenizer = processor.tokenizer
        
#         # Special tokens 추가
#         special_tokens = {
#             'additional_special_tokens': [
#                 '<coord_start>',
#                 '<coord_end>',
#                 '<x>',
#                 '<y>',
#                 '<pos>',
#                 '<neg>'
#             ]
#         }
#         num_added_tokens = tokenizer.add_special_tokens(special_tokens)
#         processor.tokenizer = tokenizer
#         logger.log(f"Added {num_added_tokens} special tokens to tokenizer")
        
#         # 모델 초기화
#         logger.log("Loading model...")
#         model_loading_config = {
#             "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
#             "rope_scaling": {"type": "dynamic", "factor": 2.0},
#             "device_map": "auto"
#         }
        
#         # Flash Attention 설정
#         if config.use_flash_attention:
#             try:
#                 import flash_attn
#                 logger.log(f"Flash Attention version: {flash_attn.__version__}")
#                 model_loading_config["attn_implementation"] = "flash_attention_2"
#             except ImportError:
#                 logger.log("Flash Attention not available", level="warning")
        
#         # 모델 로드 및 토큰 임베딩 리사이징
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             config.model_name,
#             **model_loading_config
#         )
#         model.resize_token_embeddings(len(tokenizer))
#         logger.log("Model loaded and token embeddings resized")
        
#         # LoRA 설정 및 적용
#         logger.log("Applying LoRA configuration...")
#         lora_config = LoraConfig(
#             r=config.lora_r,
#             lora_alpha=config.lora_alpha,
#             lora_dropout=config.lora_dropout,
#             target_modules=config.lora_target_modules,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         logger.log("LoRA applied to model")
        
#         # 데이터셋 초기화
#         logger.log("Initializing datasets...")
#         train_dataset = SegmentationDataset(
#             json_path=config.train_data_path,
#             processor=processor,
#             image_base_dir=config.image_base_dir,
#             image_size=config.image_size,  # 추가된 설정
#             min_valid_points=config.min_valid_points  # 설정에서 값 전달
#         )
#         val_dataset = SegmentationDataset(
#             json_path=config.val_data_path,
#             processor=processor,
#             image_base_dir=config.image_base_dir,
#             image_size=config.image_size,  # 추가된 설정
#             min_valid_points=config.min_valid_points  # 설정에서 값 전달
#         )
        
#         logger.log(f"Training samples: {len(train_dataset)}")
#         logger.log(f"Validation samples: {len(val_dataset)}")
        
#         # 데이터셋 검증
#         verify_dataset(train_dataset, logger)
#         verify_dataset(val_dataset, logger)
    
#         # 데이터로더 초기화
#         logger.log("Initializing data loaders...")
#         train_dataloader, eval_dataloader = init_data_loaders(
#             train_dataset=train_dataset,
#             val_dataset=val_dataset,
#             config=config
#         )
#         logger.log("Data loaders initialized successfully")       
        
#         # 옵티마이저 초기화
#         optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
#         # 학습률 스케줄러 설정
#         num_update_steps_per_epoch = len(train_dataloader)
#         max_train_steps = config.num_epochs * num_update_steps_per_epoch
#         num_warmup_steps = int(max_train_steps * config.warmup_ratio)
        
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=num_warmup_steps,
#             num_training_steps=max_train_steps
#         )
        
#         # Accelerator 준비
#         model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
#             model, optimizer, train_dataloader, eval_dataloader, scheduler
#         )
        
#         # 학습 실행
#         logger.log("Starting training...")
#         total_steps = 0
#         best_eval_loss = float('inf')
        
#         for epoch in range(config.num_epochs):
#             logger.log(f"Starting Epoch {epoch+1}/{config.num_epochs}")
#             total_steps, epoch_loss = train_epoch(
#                 model=model,
#                 train_dataloader=train_dataloader,
#                 eval_dataloader=eval_dataloader,
#                 optimizer=optimizer,
#                 scheduler=scheduler,
#                 accelerator=accelerator,
#                 epoch=epoch,
#                 config=config,
#                 logger=logger,
#                 total_steps=total_steps,
#                 processor=processor
#             )
            
#             # 에포크 종료 후 평가
#             eval_metrics = evaluate(
#                 model, eval_dataloader, accelerator, config, logger, 
#                 total_steps, processor
#             )
            
#             # 최고 성능 모델 저장
#             if eval_metrics.get('eval/total_loss', float('inf')) < best_eval_loss:
#                 best_eval_loss = eval_metrics['eval/total_loss']
#                 if accelerator.is_main_process:
#                     save_checkpoint(
#                         model,
#                         processor,
#                         accelerator,
#                         os.path.join(config.checkpoint_dir, "best_model"),
#                         logger
#                     )
#                     logger.log(f"New best model found at epoch {epoch+1}, loss: {best_eval_loss:.4f}")
                    
#             logger.log(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}")
        
#         logger.log("Training completed successfully.")
        
#     except Exception as e:
#         logger.log(f"Error during training: {str(e)}", level="error")
#         raise e
# def extract_coordinates_from_ids(generated_ids: torch.Tensor, tokenizer) -> Dict[str, torch.Tensor]:
#     """생성된 토큰 ID에서 좌표 추출"""
#     batch_size = generated_ids.size(0)
#     device = generated_ids.device
    
#     try:
#         texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        
#         # 결과 저장을 위한 리스트
#         positive_points_batch = []
#         negative_points_batch = []
        
#         for text in texts:
#             try:
#                 # coord_start와 coord_end 사이의 텍스트만 추출
#                 coord_pattern = r'<coord_start>(.*?)<coord_end>'
#                 coord_match = re.search(coord_pattern, text)
                
#                 if not coord_match:
#                     positive_points_batch.append(torch.zeros((1, 2), device=device))
#                     negative_points_batch.append(torch.zeros((1, 2), device=device))
#                     continue
                
#                 coord_text = coord_match.group(1)
                
#                 # 좌표 추출
#                 pos_pattern = r'<pos><x>(-?\d*\.?\d+)<y>(-?\d*\.?\d+)>'
#                 neg_pattern = r'<neg><x>(-?\d*\.?\d+)<y>(-?\d*\.?\d+)>'
                
#                 # 좌표 변환 및 정규화
#                 pos_points = [[float(x), float(y)] for x, y in re.findall(pos_pattern, coord_text)]
#                 neg_points = [[float(x), float(y)] for x, y in re.findall(neg_pattern, coord_text)]
                
#                 if not pos_points:
#                     pos_points = [[0.0, 0.0]]
#                 if not neg_points:
#                     neg_points = [[0.0, 0.0]]
                    
#                 positive_points_batch.append(torch.tensor(pos_points, device=device))
#                 negative_points_batch.append(torch.tensor(neg_points, device=device))
                
#             except Exception as e:
#                 logging.warning(f"Error processing text: {str(e)}")
#                 positive_points_batch.append(torch.zeros((1, 2), device=device))
#                 negative_points_batch.append(torch.zeros((1, 2), device=device))
        
#         # 패딩
#         max_pos = max(p.size(0) for p in positive_points_batch)
#         max_neg = max(n.size(0) for n in negative_points_batch)
        
#         padded_pos = []
#         padded_neg = []
#         pos_masks = []
#         neg_masks = []
        
#         for pos, neg in zip(positive_points_batch, negative_points_batch):
#             pos_mask = torch.ones(pos.size(0), device=device)
#             neg_mask = torch.ones(neg.size(0), device=device)
            
#             if pos.size(0) < max_pos:
#                 pad_size = max_pos - pos.size(0)
#                 pos = torch.cat([pos, torch.zeros(pad_size, 2, device=device)], dim=0)
#                 pos_mask = torch.cat([pos_mask, torch.zeros(pad_size, device=device)], dim=0)
                
#             if neg.size(0) < max_neg:
#                 pad_size = max_neg - neg.size(0)
#                 neg = torch.cat([neg, torch.zeros(pad_size, 2, device=device)], dim=0)
#                 neg_mask = torch.cat([neg_mask, torch.zeros(pad_size, device=device)], dim=0)
            
#             padded_pos.append(pos)
#             padded_neg.append(neg)
#             pos_masks.append(pos_mask)
#             neg_masks.append(neg_mask)
        
#         return {
#             'positive_points': torch.stack(padded_pos),
#             'negative_points': torch.stack(padded_neg),
#             'positive_mask': torch.stack(pos_masks),
#             'negative_mask': torch.stack(neg_masks)
#         }
        
#     except Exception as e:
#         logging.error(f"Fatal error in coordinate extraction: {str(e)}")
#         return {
#             'positive_points': torch.zeros((batch_size, 1, 2), device=device),
#             'negative_points': torch.zeros((batch_size, 1, 2), device=device),
#             'positive_mask': torch.zeros((batch_size, 1), device=device),
#             'negative_mask': torch.zeros((batch_size, 1), device=device)
#         }

# def init_data_loaders(train_dataset: SegmentationDataset, val_dataset: SegmentationDataset, config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
#     """데이터로더 초기화"""
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config.train_batch_size,
#         shuffle=True,
#         num_workers=0,  # 멀티프로세싱 비활성화
#         collate_fn=SegmentationDataset.collate_fn,
#         pin_memory=True
#     )
    
#     eval_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config.eval_batch_size,
#         shuffle=False,
#         num_workers=0,  # 멀티프로세싱 비활성화
#         collate_fn=SegmentationDataset.collate_fn,
#         pin_memory=True
#     )
    
#     return train_dataloader, eval_dataloader

# def train_epoch(
#     model: torch.nn.Module,
#     train_dataloader: DataLoader,
#     eval_dataloader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LambdaLR,
#     accelerator: Accelerator,
#     epoch: int,
#     config: TrainingConfig,
#     logger: EnhancedLogger,
#     total_steps: int,
#     processor: AutoProcessor
# ) -> Tuple[int, float]:
#     """한 에포크 동안의 학습"""
#     model.train()
#     epoch_loss = 0.0
#     progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", disable=not accelerator.is_local_main_process)
    
#     for batch in progress_bar:
#         with accelerator.accumulate(model):
#             try:
#                 # 모델 입력 준비
#                 model_inputs = {
#                     'input_ids': batch['input_ids'],
#                     'attention_mask': batch['attention_mask'],
#                     'labels': batch['labels'],
#                     'pixel_values': batch['pixel_values'],
#                     'positive_points': batch['positive_points'],
#                     'negative_points': batch['negative_points'],
#                     'positive_mask': batch['positive_mask'],
#                     'negative_mask': batch['negative_mask'],
#                     'original_size': batch['original_size'],
#                     'target_size': batch['target_size']
#                 }
                
#                 outputs = model(**model_inputs)
#                 lm_loss = outputs.loss
                
#                 # 좌표 손실 계산
#                 generated_ids = accelerator.unwrap_model(model).generate(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     pixel_values=batch['pixel_values'],
#                     max_length=config.max_length,
#                     num_beams=4,
#                     do_sample=False
#                 )
                
#                 pred_coordinates = extract_coordinates_from_ids(
#                     generated_ids,
#                     processor.tokenizer
#                 )
                
#                 point_loss_fn = SegmentationLoss(
#                     positive_weight=config.positive_weight,
#                     negative_weight=config.negative_weight,
#                     sigma=config.sigma,
#                     min_valid_points=config.min_valid_points
#                 ).to(accelerator.device)
                
#                 point_loss_dict = point_loss_fn(
#                     pred_coordinates=pred_coordinates,
#                     target={
#                         'positive_points': batch['positive_points'],
#                         'negative_points': batch['negative_points']
#                     },
#                     masks={
#                         'positive_mask': batch.get('positive_mask'),
#                         'negative_mask': batch.get('negative_mask')
#                     }
#                 )
                
#                 total_loss = lm_loss + config.coordinate_loss_weight * point_loss_dict['loss']
                
#                 # 그래디언트 계산 및 옵티마이저 스텝
#                 accelerator.backward(total_loss)
                
#                 if config.max_grad_norm > 0:
#                     accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()
                
#                 epoch_loss += total_loss.item()
#                 progress_bar.set_postfix(loss=total_loss.item())
#                 total_steps += 1
                
#                 if total_steps % config.logging_steps == 0:
#                     logger.log_metrics({
#                         'train/total_loss': total_loss.item(),
#                         'train/lm_loss': lm_loss.item(),
#                         'train/pos_loss': point_loss_dict['pos_loss'],
#                         'train/neg_loss': point_loss_dict['neg_loss'],
#                         'train/spread_loss': point_loss_dict['spread_loss']
#                     }, total_steps)
                    
#             except Exception as e:
#                 logger.log(f"Error processing batch: {str(e)}", level="error")
#                 continue
        
#     average_loss = epoch_loss / len(train_dataloader)
#     return total_steps, average_loss

# def evaluate(
#     model: torch.nn.Module,
#     eval_dataloader: DataLoader,
#     accelerator: Accelerator,
#     config: TrainingConfig,
#     logger: EnhancedLogger,
#     total_steps: int,
#     processor: AutoProcessor
# ) -> Dict[str, float]:
#     """평가 수행"""
#     model.eval()
#     eval_loss = 0.0
#     total_lm_loss = 0.0
#     total_point_loss = 0.0
#     all_metrics = []
    
#     point_loss_fn = SegmentationLoss(
#         positive_weight=config.positive_weight,
#         negative_weight=config.negative_weight,
#         sigma=config.sigma,
#         min_valid_points=config.min_valid_points
#     ).to(accelerator.device)
    
#     with torch.no_grad():
#         for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
#             try:
#                 # 모델 입력 준비
#                 model_inputs = {
#                     'input_ids': batch['input_ids'],
#                     'attention_mask': batch['attention_mask'],
#                     'labels': batch['labels'],
#                     'pixel_values': batch['pixel_values'],
#                     'positive_points': batch['positive_points'],
#                     'negative_points': batch['negative_points'],
#                     'positive_mask': batch['positive_mask'],
#                     'negative_mask': batch['negative_mask'],
#                     'original_size': batch['original_size'],
#                     'target_size': batch['target_size']
#                 }
                
#                 outputs = model(**model_inputs)
#                 lm_loss = outputs.loss
                
#                 # 좌표 손실 계산
#                 generated_ids = accelerator.unwrap_model(model).generate(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'],
#                     pixel_values=batch['pixel_values'],
#                     max_length=config.max_length,
#                     num_beams=4,
#                     do_sample=False
#                 )
                
#                 pred_coordinates = extract_coordinates_from_ids(
#                     generated_ids,
#                     processor.tokenizer
#                 )
                
#                 point_loss_dict = point_loss_fn(
#                     pred_coordinates=pred_coordinates,
#                     target={
#                         'positive_points': batch['positive_points'],
#                         'negative_points': batch['negative_points']
#                     },
#                     masks={
#                         'positive_mask': batch.get('positive_mask'),
#                         'negative_mask': batch.get('negative_mask')
#                     }
#                 )
                
#                 total_loss = lm_loss + config.coordinate_loss_weight * point_loss_dict['loss']
                
#                 # 손실 누적
#                 eval_loss += total_loss.item()
#                 total_lm_loss += lm_loss.item()
#                 total_point_loss += point_loss_dict['loss'].item()
                
#                 # 메트릭 수집
#                 metrics = {
#                     'eval/total_loss': total_loss.item(),
#                     'eval/lm_loss': lm_loss.item(),
#                     'eval/point_loss': point_loss_dict['loss'].item(),
#                     'eval/pos_loss': point_loss_dict['pos_loss'],
#                     'eval/neg_loss': point_loss_dict['neg_loss'],
#                     'eval/spread_loss': point_loss_dict['spread_loss']
#                 }
#                 all_metrics.append(metrics)
                
#             except Exception as e:
#                 logger.log(f"Error in evaluation step: {str(e)}", level="error")
#                 continue
    
#     num_batches = len(eval_dataloader)
#     if num_batches == 0:
#         logger.log("No evaluation batches available.", level="warning")
#         avg_metrics = {}
#     else:
#         avg_metrics = {
#             'eval/total_loss': eval_loss / num_batches,
#             'eval/lm_loss': total_lm_loss / num_batches,
#             'eval/point_loss': total_point_loss / num_batches,
#             'eval/step': total_steps
#         }
        
#         # 세부 메트릭 평균 계산
#         for key in ['pos_loss', 'neg_loss', 'spread_loss']:
#             avg_metrics[f'eval/{key}'] = sum(m[key] for m in all_metrics) / num_batches
    
#     logger.log_metrics(avg_metrics, total_steps)
    
#     return avg_metrics

# def save_checkpoint(
#     model: torch.nn.Module,
#     processor: AutoProcessor,
#     accelerator: Accelerator,
#     checkpoint_path: str,
#     logger: EnhancedLogger
# ):
#     """체크포인트 저장"""
#     try:
#         # 저장 경로 생성
#         save_path = Path(checkpoint_path)
#         save_path.mkdir(parents=True, exist_ok=True)
        
#         # 모델 저장
#         unwrapped_model = accelerator.unwrap_model(model)
#         unwrapped_model.save_pretrained(
#             save_path,
#             save_function=accelerator.save,
#             is_main_process=accelerator.is_main_process
#         )
        
#         # 프로세서 저장
#         if accelerator.is_main_process:
#             processor.save_pretrained(save_path)
            
#         logger.log(f"Checkpoint saved to {save_path}")
        
#     except Exception as e:
#         logger.log(f"Error saving checkpoint: {str(e)}", level="error")

# def verify_dataset(dataset: SegmentationDataset, logger: EnhancedLogger):
#     """데이터셋 검증"""
#     missing_labels = []
#     for idx in range(len(dataset)):
#         try:
#             sample = dataset[idx]
#             if 'labels' not in sample:
#                 missing_labels.append(idx)
#         except Exception as e:
#             logger.log(f"Error accessing sample {idx}: {str(e)}", level="error")
#             missing_labels.append(idx)
#     if missing_labels:
#         logger.log(f"Samples missing 'labels' at indices: {missing_labels}", level="error")
#     else:
#         logger.log("All samples contain 'labels' key.", level="info")

# if __name__ == "__main__":
#     try:
#         config = TrainingConfig(
#             lora_r=8,
#             lora_alpha=16,
#             lora_dropout=0.1,
#             lora_target_modules=["q_proj", "v_proj"],
#             learning_rate=1e-4,
#             num_epochs=10,
#             train_batch_size=4,
#             eval_batch_size=4,
#             gradient_accumulation_steps=2,
#             warmup_ratio=0.1,  # warmup_steps를 warmup_ratio로 변경
#             max_grad_norm=1.0,
#             save_steps=500,
#             eval_steps=100,
#             logging_steps=10,  # 추가된 설정
#             use_wandb=True
#         )
        
#         train(config)
#     except Exception as e:
#         logging.error(f"Training failed: {str(e)}")
#         raise
    