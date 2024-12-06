import os
import cv2
import numpy as np
import torch
from algorithm.core import Qwen2V_SAM2_Connector,LLaVA_Next_SAM2_Connector,OpanAIConnector

class BatchImageProcessor:
    def __init__(self, input_dir, output_dir, prompt="segment everything"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prompt = prompt
        self.connector = Qwen2V_SAM2_Connector()
        self.image_size = (512, 512)
        
        # 기본 디렉토리 생성
        self.resized_dir = os.path.join(output_dir, "resized")
        
        # Individual masks 디렉토리 구조
        self.individual_dir = os.path.join(output_dir, "individual")
        self.individual_masks_dir = os.path.join(self.individual_dir, "masks")
        self.individual_overlays_dir = os.path.join(self.individual_dir, "overlays")
        
        # Combined masks 디렉토리 구조
        self.combined_dir = os.path.join(output_dir, "combined")
        self.combined_masks_dir = os.path.join(self.combined_dir, "masks")
        self.combined_overlays_dir = os.path.join(self.combined_dir, "overlays")
        
        # 모든 디렉토리 생성
        os.makedirs(self.resized_dir, exist_ok=True)
        os.makedirs(self.individual_masks_dir, exist_ok=True)
        os.makedirs(self.individual_overlays_dir, exist_ok=True)
        os.makedirs(self.combined_masks_dir, exist_ok=True)
        os.makedirs(self.combined_overlays_dir, exist_ok=True)

    def resize_image(self, image):
        """이미지를 지정된 크기로 리사이즈"""
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def draw_grid(self, image):
        """3x3 흰색 점선 그리드 그리기"""
        height, width = image.shape[:2]
        cell_height = height // 3
        cell_width = width // 3
        
        grid_image = image.copy()
        
        for i in range(1, 3):
            x = i * cell_width
            cv2.line(grid_image, (x, 0), (x, height), (255, 255, 255), 1, lineType=cv2.LINE_AA, shift=0)
        
        for i in range(1, 3):
            y = i * cell_height
            cv2.line(grid_image, (0, y), (width, y), (255, 255, 255), 1, lineType=cv2.LINE_AA, shift=0)
            
        return grid_image

    def combine_masks(self, masks):
        """여러 마스크를 하나의 마스크로 결합"""
        if not masks:
            return None
            
        combined = np.zeros(self.image_size, dtype=np.uint8)
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            mask = np.squeeze(mask)
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            
            if mask.shape != self.image_size:
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            combined = cv2.max(combined, mask)
        
        return combined

    def process_single_image(self, image_path):
        try:
            # 이미지 읽기 및 전처리
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return
            
            resized_image = self.resize_image(image)
            grid_image = self.draw_grid(resized_image)
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # 마스크 생성
            masks = self.connector.process_image_text(image_rgb, self.prompt)
            
            if not masks or len(masks) == 0:
                print(f"No masks detected for: {image_path}")
                return

            # 파일 이름 설정
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 리사이즈된 원본 이미지 저장
            resized_path = os.path.join(self.resized_dir, f"{base_name}_resized.jpg")
            cv2.imwrite(resized_path, grid_image)
            
            # 개별 마스크 처리 및 저장
            processed_masks = []
            for idx, mask_logit in enumerate(masks):
                # 마스크 처리
                if isinstance(mask_logit, torch.Tensor):
                    mask = mask_logit.cpu().numpy()
                else:
                    mask = mask_logit
                
                mask = np.squeeze(mask)
                mask = (mask * 255).astype(np.uint8)
                
                if mask.shape != self.image_size:
                    mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                
                processed_masks.append(mask)
                
                # 개별 마스크 저장 (individual/masks 디렉토리)
                mask_path = os.path.join(self.individual_masks_dir, f"{base_name}_mask_{idx}.png")
                cv2.imwrite(mask_path, mask)
                
                # 개별 오버레이 생성 및 저장 (individual/overlays 디렉토리)
                colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                overlay = grid_image.copy()
                overlay = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)
                overlay_path = os.path.join(self.individual_overlays_dir, f"{base_name}_overlay_{idx}.jpg")
                cv2.imwrite(overlay_path, overlay)
            
            # 결합된 마스크 생성 및 저장
            combined_mask = self.combine_masks(processed_masks)
            if combined_mask is not None:
                # 결합된 마스크 저장 (combined/masks 디렉토리)
                combined_mask_path = os.path.join(self.combined_masks_dir, f"{base_name}_combined_mask.png")
                cv2.imwrite(combined_mask_path, combined_mask)
                
                # 결합된 마스크의 오버레이 생성 및 저장 (combined/overlays 디렉토리)
                colored_combined = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
                combined_overlay = grid_image.copy()
                combined_overlay = cv2.addWeighted(combined_overlay, 0.5, colored_combined, 0.5, 0)
                combined_overlay_path = os.path.join(self.combined_overlays_dir, f"{base_name}_combined_overlay.jpg")
                cv2.imwrite(combined_overlay_path, combined_overlay)
                
                print(f"Saved all masks and overlays for {base_name}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    def process_directory(self, max_images=50):
        """디렉토리 내의 모든 이미지 처리 (최대 max_images개 처리)"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        processed_count = 0
        error_count = 0
        
        print(f"Starting batch processing from: {self.input_dir}")
        print(f"Images will be resized to: {self.image_size}")
        
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, self.input_dir)
                    
                    # 상대 경로에 맞는 출력 디렉토리 생성
                    if relative_path != ".":
                        for dir_path in [self.resized_dir, 
                                    self.individual_masks_dir, 
                                    self.individual_overlays_dir,
                                    self.combined_masks_dir,
                                    self.combined_overlays_dir]:
                            current_dir = os.path.join(dir_path, relative_path)
                            os.makedirs(current_dir, exist_ok=True)
                    
                    try:
                        print(f"Processing: {image_path}")
                        self.process_single_image(image_path)
                        processed_count += 1
                        
                        # Stop processing after reaching max_images
                        if processed_count >= max_images:
                            print("Reached the maximum number of images to process.")
                            return
                    
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
                        error_count += 1
        
        print(f"\nProcessing completed!")
        print(f"Total images processed: {processed_count}")
        print(f"Errors encountered: {error_count}")
        print(f"Results saved in:")
        print(f"- Resized images: {self.resized_dir}")
        print(f"- Individual masks: {self.individual_masks_dir}")
        print(f"- Individual overlays: {self.individual_overlays_dir}")
        print(f"- Combined masks: {self.combined_masks_dir}")
        print(f"- Combined overlays: {self.combined_overlays_dir}")


def main():
    input_dir = "/home/joongwon00/coco_dataset/person_images"
    output_dir = "./LLAVA_coco_output"
    prompt = "detect every person."
    
    processor = BatchImageProcessor(input_dir, output_dir, prompt)
    processor.process_directory()

if __name__ == "__main__":
    main()
