"""
OCR Pipeline — YOLOv8 Detection + CRNN Recognition

Pipeline:
    1. YOLOv8 mendeteksi lokasi kata dalam gambar
    2. Crop setiap region yang terdeteksi
    3. CRNN mengenali karakter dari setiap crop
    4. CTC Decoding menghasilkan transliterasi Latin
"""

import os
import time
import functools
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ============================================================
# Fix PyTorch 2.6+ weights_only=True default
# Ultralytics torch_safe_load calls torch.load without weights_only=False,
# causing UnpicklingError. We patch torch.load to default weights_only=False.
# ============================================================
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO

from models import CRNNHybrid


class CTCLabelConverter:
    """
    Converter untuk CTC: Encode (training) dan Decode (inference)
    Index 0 = CTC BLANK token
    Index 1-N = karakter dari charset
    """

    def __init__(self, charset: str):
        self.charset = charset
        self.blank_idx = 0

        self.char_to_idx = {}
        self.idx_to_char = {0: ''}

        for idx, char in enumerate(charset):
            self.char_to_idx[char] = idx + 1
            self.idx_to_char[idx + 1] = char

        self.num_classes = len(charset) + 1

    def decode(self, indices, raw: bool = False) -> str:
        """
        Decode indices menjadi text (greedy decoding)
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()

        if raw:
            chars = []
            for idx in indices:
                if idx == 0:
                    chars.append('-')
                else:
                    chars.append(self.idx_to_char.get(idx, '?'))
            return ''.join(chars)

        # Collapse: hapus blank dan karakter berulang
        decoded = []
        prev_idx = -1

        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx != self.blank_idx and idx != prev_idx:
                char = self.idx_to_char.get(idx, '')
                if char:
                    decoded.append(char)
            prev_idx = idx

        return ''.join(decoded)

    def get_info(self) -> dict:
        return {
            'charset': self.charset,
            'num_chars': len(self.charset),
            'num_classes': self.num_classes
        }


class OCRPipeline:
    """
    Pipeline OCR lengkap untuk aksara Jawa
    Stage 1: YOLOv8 untuk deteksi kata
    Stage 2: CRNN untuk pengenalan karakter
    """

    def __init__(
        self,
        yolo_model_path: str,
        crnn_model_path: str,
        charset_path: str,
        device: str = 'cpu',
        yolo_conf: float = 0.25,
        crnn_img_size: Tuple[int, int] = (32, 128)
    ):
        self.device = torch.device(device)
        self.yolo_conf = yolo_conf
        self.crnn_img_height, self.crnn_img_width = crnn_img_size

        # Load charset
        with open(charset_path, 'r', encoding='utf-8') as f:
            self.charset = f.read().strip()

        # Initialize CTC converter
        self.converter = CTCLabelConverter(self.charset)

        # Load YOLOv8 model
        self.yolo_model = YOLO(yolo_model_path)

        # Load CRNN model
        checkpoint = torch.load(crnn_model_path, map_location=self.device, weights_only=False)

        self.crnn_model = CRNNHybrid(
            num_classes=self.converter.num_classes,
            pretrained_vgg=False,
            freeze_vgg=True
        )
        self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.crnn_model.to(self.device)
        self.crnn_model.eval()

        # CRNN preprocessing transform
        self.crnn_transform = transforms.Compose([
            transforms.Resize((self.crnn_img_height, self.crnn_img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_words(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Stage 1: Deteksi kata menggunakan YOLOv8

        Returns:
            Tuple of (list of detections, detection time in ms)
        """
        start_time = time.time()

        results = self.yolo_model(image, conf=self.yolo_conf, verbose=False)

        detection_time = (time.time() - start_time) * 1000

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                # Crop region
                crop = image[y1:y2, x1:x2]

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'crop': crop
                })

        # Sort by y first (top to bottom), then x (left to right) for reading order
        detections.sort(key=lambda d: (d['bbox'][1], d['bbox'][0]))

        return detections, detection_time

    def recognize_text(self, crop: np.ndarray) -> Dict:
        """
        Stage 2: Pengenalan teks menggunakan CRNN

        Returns:
            Dict dengan keys: text, confidence, raw_output, time_ms
        """
        start_time = time.time()

        # Convert to PIL Image
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(crop)

        # Preprocess
        tensor = self.crnn_transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.crnn_model(tensor)
            probs = torch.softmax(output, dim=2)
            max_probs, indices = probs.max(dim=2)

            # Calculate confidence
            non_blank_mask = indices[0] != 0
            if non_blank_mask.any():
                confidence = float(max_probs[0][non_blank_mask].mean().cpu().numpy())
            else:
                confidence = 0.0

        recognition_time = (time.time() - start_time) * 1000

        # Decode
        raw_output = self.converter.decode(indices[0], raw=True)
        text = self.converter.decode(indices[0])

        return {
            'text': text,
            'confidence': confidence,
            'raw_output': raw_output,
            'time_ms': recognition_time
        }

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Proses gambar lengkap: deteksi + pengenalan

        Args:
            image: numpy array (BGR format from cv2)

        Returns:
            Dict dengan hasil lengkap
        """
        total_start = time.time()

        # Stage 1: Detection
        detections, detection_time = self.detect_words(image)

        # Stage 2: Recognition
        results = []
        total_recognition_time = 0

        for det in detections:
            if det['crop'].size > 0:
                recognition = self.recognize_text(det['crop'])
                total_recognition_time += recognition['time_ms']

                results.append({
                    'bbox': det['bbox'],
                    'detection_conf': det['confidence'],
                    'text': recognition['text'],
                    'recognition_conf': recognition['confidence'],
                    'raw_output': recognition['raw_output'],
                    'crop': det['crop']
                })

        total_time = (time.time() - total_start) * 1000

        return {
            'image_shape': image.shape,
            'num_detections': len(results),
            'results': results,
            'timing': {
                'detection_ms': detection_time,
                'recognition_ms': total_recognition_time,
                'total_ms': total_time
            },
            'full_text': ' '.join([r['text'] for r in results])
        }
