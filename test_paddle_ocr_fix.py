"""
Test script to verify PaddleOCR with the updated API
This demonstrates how to fix the PaddleOCR integration
"""

import cv2
import os
from pathlib import Path
from paddleocr import PaddleOCR
import numpy as np


def test_paddle_ocr_updated():
    """Test PaddleOCR with updated API (predict method)."""
    print("="*60)
    print("Testing PaddleOCR with Updated API")
    print("="*60)

    # Initialize PaddleOCR with new parameters
    ocr = PaddleOCR(
        use_textline_orientation=True,  # New parameter name
        lang='en'
    )

    # Test data path
    test_path = Path(r"C:\Users\Rishichowdary-3925\Downloads\No plates")

    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(test_path.glob(ext)))

    print(f"\nFound {len(test_images)} test images")
    print("\nProcessing images...\n")

    # Test on each image
    results_summary = []

    for img_path in test_images:
        print(f"\n--- Processing: {img_path.name} ---")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print("  [FAIL] Failed to read image")
            continue

        # Convert BGR to RGB for PaddleOCR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # Use the updated predict() method instead of ocr()
            # Note: predict() may return different format than ocr()
            # Let's try the standard ocr() first without cls parameter
            results = ocr.ocr(img_rgb)

            if results and results[0]:
                print(f"  [OK] OCR Results:")

                texts = []
                confidences = []

                for line in results[0]:
                    if isinstance(line, list) and len(line) >= 2:
                        # Format: [bbox, (text, confidence)]
                        bbox = line[0]
                        text_info = line[1]

                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            conf = text_info[1]

                            texts.append(text)
                            confidences.append(conf)

                            print(f"    Text: '{text}' (Confidence: {conf:.3f})")

                if texts:
                    combined_text = ' '.join(texts)
                    avg_conf = np.mean(confidences)

                    results_summary.append({
                        'image': img_path.name,
                        'text': combined_text,
                        'confidence': avg_conf
                    })

                    print(f"  -> Combined: '{combined_text}' (Avg Conf: {avg_conf:.3f})")
            else:
                print("  X No text detected")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results_summary:
        print(f"\nSuccessfully processed {len(results_summary)}/{len(test_images)} images\n")

        print("Top Results:")
        # Sort by confidence
        sorted_results = sorted(results_summary, key=lambda x: x['confidence'], reverse=True)

        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i}. {result['image']}")
            print(f"   Text: {result['text']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print()

    print("="*60)


def test_on_detected_plates():
    """Test OCR on detected plates using YOLO."""
    print("\n" + "="*60)
    print("Testing PaddleOCR on Detected Plates")
    print("="*60)

    from ultralytics import YOLO

    # Load detector
    detector = YOLO('yolov8n.pt')
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    # Test path
    test_path = Path(r"C:\Users\Rishichowdary-3925\Downloads\No plates")
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(test_path.glob(ext)))

    print(f"\nTesting on {len(test_images)} images\n")

    total_plates_detected = 0
    total_plates_read = 0
    all_results = []

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Detect plates
        det_results = detector(img, conf=0.3, verbose=False)

        for result in det_results:
            for box in result.boxes:
                total_plates_detected += 1

                # Extract plate region
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Add padding
                h_img, w_img = img.shape[:2]
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * 0.2)
                pad_y = int(box_h * 0.05)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w_img, x2 + pad_x)
                y2 = min(h_img, y2 + pad_y)

                plate_img = img[y1:y2, x1:x2]
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

                # Run OCR
                try:
                    ocr_results = ocr.ocr(plate_rgb)

                    if ocr_results and ocr_results[0]:
                        texts = []
                        for line in ocr_results[0]:
                            if isinstance(line, list) and len(line) >= 2:
                                text_info = line[1]
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    texts.append(text_info[0])

                        if texts:
                            total_plates_read += 1
                            combined = ' '.join(texts)
                            all_results.append({
                                'image': img_path.name,
                                'text': combined,
                                'det_conf': float(box.conf[0])
                            })

                            print(f"[OK] {img_path.name}: {combined} (Det: {box.conf[0]:.2f})")

                except Exception as e:
                    print(f"[ERROR] OCR Error on {img_path.name}: {e}")

    # Summary
    print("\n" + "="*60)
    print(f"Plates Detected: {total_plates_detected}")
    print(f"Plates Read Successfully: {total_plates_read}")
    print(f"OCR Success Rate: {(total_plates_read/total_plates_detected)*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    # Test 1: Direct image OCR
    test_paddle_ocr_updated()

    # Test 2: Detection + OCR pipeline
    test_on_detected_plates()
