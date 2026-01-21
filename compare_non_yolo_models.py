"""
Non-YOLO Detection Model Comparison
Compare Custom-Plate-Detector vs other popular detection frameworks:
- Faster R-CNN (Detectron2)
- RetinaNet (Detectron2)
- SSD (TensorFlow)
- EfficientDet (TensorFlow)
- CenterNet (PyTorch)
- DETR (Transformers)
"""

import os
import sys
import cv2
import time
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Import existing services
sys.path.append(os.path.dirname(__file__))
from services.plate_ocr import PlateOCR


class NonYOLOComparison:
    """Compare non-YOLO models for plate detection."""

    def __init__(self, test_data_path):
        """Initialize with test data."""
        self.test_data_path = Path(test_data_path)
        self.ocr = PlateOCR()

        # Get test images
        self.test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.test_images.extend(list(self.test_data_path.glob(ext)))

        print(f"Found {len(self.test_images)} test images")
        self.results = {}

    def test_custom_model(self):
        """Test custom plate detector (baseline)."""
        print(f"\n{'='*60}")
        print(f"Testing: Custom-Plate-Detector (Baseline)")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLO
            detector = YOLO('license_plate_detector.pt')

            total_time_det = 0
            total_time_ocr = 0
            total_detections = 0
            successful_reads = 0
            det_confs = []
            ocr_confs = []
            results = []

            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Detection
                start = time.time()
                det_results = detector(img, conf=0.3, verbose=False)
                det_time = time.time() - start
                total_time_det += det_time

                for result in det_results:
                    for box in result.boxes:
                        total_detections += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        det_confs.append(conf)

                        # Crop with padding
                        h, w = img.shape[:2]
                        pad_x = int((x2-x1) * 0.2)
                        pad_y = int((y2-y1) * 0.05)
                        x1 = max(0, x1-pad_x)
                        y1 = max(0, y1-pad_y)
                        x2 = min(w, x2+pad_x)
                        y2 = min(h, y2+pad_y)
                        plate = img[y1:y2, x1:x2]

                        # OCR
                        start = time.time()
                        ocr_result = self.ocr.read_plate(plate)
                        ocr_time = time.time() - start
                        total_time_ocr += ocr_time

                        text = ocr_result.get('text', '')
                        ocr_conf = ocr_result.get('confidence', 0)

                        if text and len(text) > 3:
                            successful_reads += 1
                            ocr_confs.append(ocr_conf)
                            results.append({
                                'image': img_path.name,
                                'text': text,
                                'det_conf': conf,
                                'ocr_conf': ocr_conf
                            })
                            print(f"  [OK] {img_path.name}: {text} (Det:{conf:.2f}, OCR:{ocr_conf:.2f})")

            metrics = {
                'total_detections': total_detections,
                'successful_reads': successful_reads,
                'avg_det_time_ms': (total_time_det/len(self.test_images))*1000,
                'avg_ocr_time_ms': (total_time_ocr/total_detections)*1000 if total_detections > 0 else 0,
                'avg_det_conf': np.mean(det_confs) if det_confs else 0,
                'avg_ocr_conf': np.mean(ocr_confs) if ocr_confs else 0,
                'ocr_success_rate': (successful_reads/total_detections)*100 if total_detections > 0 else 0,
                'sample_results': results[:10]
            }

            self.results['Custom-Plate-Detector'] = metrics
            print(f"\nSummary: {successful_reads}/{total_detections} plates read successfully")
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def test_faster_rcnn(self):
        """Test Faster R-CNN from Detectron2."""
        print(f"\n{'='*60}")
        print(f"Testing: Faster R-CNN (Detectron2)")
        print(f"{'='*60}")

        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo

            # Configure Faster R-CNN
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = "cpu"
            predictor = DefaultPredictor(cfg)

            total_time_det = 0
            total_time_ocr = 0
            total_detections = 0
            successful_reads = 0
            det_confs = []
            ocr_confs = []
            results = []

            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Detection
                start = time.time()
                outputs = predictor(img)
                det_time = time.time() - start
                total_time_det += det_time

                # Filter for relevant classes (car, truck, etc.)
                instances = outputs["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()

                # COCO classes: 2=car, 7=truck, 5=bus
                vehicle_classes = [2, 5, 7]

                for box, score, cls in zip(boxes, scores, classes):
                    if cls in vehicle_classes and score > 0.3:
                        total_detections += 1
                        x1, y1, x2, y2 = map(int, box)
                        det_confs.append(float(score))

                        # Crop plate region (assuming plate is in lower portion)
                        h, w = img.shape[:2]
                        y1_plate = int(y1 + (y2-y1)*0.6)
                        plate = img[y1_plate:y2, x1:x2]

                        if plate.size == 0:
                            continue

                        # OCR
                        start = time.time()
                        ocr_result = self.ocr.read_plate(plate)
                        ocr_time = time.time() - start
                        total_time_ocr += ocr_time

                        text = ocr_result.get('text', '')
                        ocr_conf = ocr_result.get('confidence', 0)

                        if text and len(text) > 3:
                            successful_reads += 1
                            ocr_confs.append(ocr_conf)
                            results.append({
                                'image': img_path.name,
                                'text': text,
                                'det_conf': float(score),
                                'ocr_conf': ocr_conf
                            })
                            print(f"  [OK] {img_path.name}: {text} (Det:{score:.2f})")

            metrics = {
                'total_detections': total_detections,
                'successful_reads': successful_reads,
                'avg_det_time_ms': (total_time_det/len(self.test_images))*1000,
                'avg_ocr_time_ms': (total_time_ocr/total_detections)*1000 if total_detections > 0 else 0,
                'avg_det_conf': np.mean(det_confs) if det_confs else 0,
                'avg_ocr_conf': np.mean(ocr_confs) if ocr_confs else 0,
                'ocr_success_rate': (successful_reads/total_detections)*100 if total_detections > 0 else 0,
                'sample_results': results[:10]
            }

            self.results['Faster R-CNN'] = metrics
            print(f"\nSummary: {successful_reads}/{total_detections} plates read successfully")
            return metrics

        except ImportError:
            print("[SKIP] Detectron2 not installed")
            print("Install: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_ssd_mobilenet(self):
        """Test SSD MobileNet from TensorFlow."""
        print(f"\n{'='*60}")
        print(f"Testing: SSD MobileNet V2 (TensorFlow)")
        print(f"{'='*60}")

        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            # Load SSD MobileNet V2 from TF Hub
            model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
            print("Loading SSD MobileNet V2...")
            detector = hub.load(model_url)
            print("Model loaded successfully")

            total_time_det = 0
            total_time_ocr = 0
            total_detections = 0
            successful_reads = 0
            det_confs = []
            ocr_confs = []
            results = []

            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_tensor = tf.convert_to_tensor(img_rgb)
                input_tensor = input_tensor[tf.newaxis, ...]

                # Detection
                start = time.time()
                detections = detector(input_tensor)
                det_time = time.time() - start
                total_time_det += det_time

                # Extract results
                boxes = detections['detection_boxes'][0].numpy()
                scores = detections['detection_scores'][0].numpy()
                classes = detections['detection_classes'][0].numpy()

                h, w = img.shape[:2]

                # COCO classes: 3=car, 8=truck, 6=bus
                vehicle_classes = [3, 6, 8]

                for box, score, cls in zip(boxes, scores, classes):
                    if int(cls) in vehicle_classes and score > 0.3:
                        total_detections += 1
                        ymin, xmin, ymax, xmax = box
                        x1, y1, x2, y2 = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)
                        det_confs.append(float(score))

                        # Crop plate region (lower portion)
                        y1_plate = int(y1 + (y2-y1)*0.6)
                        plate = img[y1_plate:y2, x1:x2]

                        if plate.size == 0:
                            continue

                        # OCR
                        start = time.time()
                        ocr_result = self.ocr.read_plate(plate)
                        ocr_time = time.time() - start
                        total_time_ocr += ocr_time

                        text = ocr_result.get('text', '')
                        ocr_conf = ocr_result.get('confidence', 0)

                        if text and len(text) > 3:
                            successful_reads += 1
                            ocr_confs.append(ocr_conf)
                            results.append({
                                'image': img_path.name,
                                'text': text,
                                'det_conf': float(score),
                                'ocr_conf': ocr_conf
                            })
                            print(f"  [OK] {img_path.name}: {text} (Det:{score:.2f})")

            metrics = {
                'total_detections': total_detections,
                'successful_reads': successful_reads,
                'avg_det_time_ms': (total_time_det/len(self.test_images))*1000,
                'avg_ocr_time_ms': (total_time_ocr/total_detections)*1000 if total_detections > 0 else 0,
                'avg_det_conf': np.mean(det_confs) if det_confs else 0,
                'avg_ocr_conf': np.mean(ocr_confs) if ocr_confs else 0,
                'ocr_success_rate': (successful_reads/total_detections)*100 if total_detections > 0 else 0,
                'sample_results': results[:10]
            }

            self.results['SSD MobileNet V2'] = metrics
            print(f"\nSummary: {successful_reads}/{total_detections} plates read successfully")
            return metrics

        except ImportError:
            print("[SKIP] TensorFlow or TensorFlow Hub not installed")
            print("Install: pip install tensorflow tensorflow-hub")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_opencv_dnn(self):
        """Test OpenCV DNN module with various models."""
        print(f"\n{'='*60}")
        print(f"Testing: OpenCV DNN (MobileNet-SSD)")
        print(f"{'='*60}")

        try:
            # Try to load pre-trained MobileNet-SSD
            prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
            model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

            prototxt_path = "mobilenet_ssd.prototxt"
            model_path = "mobilenet_ssd.caffemodel"

            # Download if not exists
            if not os.path.exists(prototxt_path):
                print("Downloading prototxt...")
                import urllib.request
                urllib.request.urlretrieve(prototxt_url, prototxt_path)

            if not os.path.exists(model_path):
                print("Downloading model weights (23MB)...")
                import urllib.request
                urllib.request.urlretrieve(model_url, model_path)

            # Load network
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

            total_time_det = 0
            total_time_ocr = 0
            total_detections = 0
            successful_reads = 0
            det_confs = []
            ocr_confs = []
            results = []

            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Prepare input blob
                blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5))

                # Detection
                start = time.time()
                net.setInput(blob)
                detections = net.forward()
                det_time = time.time() - start
                total_time_det += det_time

                # Process detections
                # Classes: 7=car, 15=person, etc.
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    class_id = int(detections[0, 0, i, 1])

                    # 7 = car in MobileNet-SSD
                    if class_id == 7 and confidence > 0.3:
                        total_detections += 1
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        det_confs.append(float(confidence))

                        # Crop plate region
                        y1_plate = int(y1 + (y2-y1)*0.6)
                        plate = img[y1_plate:y2, x1:x2]

                        if plate.size == 0:
                            continue

                        # OCR
                        start = time.time()
                        ocr_result = self.ocr.read_plate(plate)
                        ocr_time = time.time() - start
                        total_time_ocr += ocr_time

                        text = ocr_result.get('text', '')
                        ocr_conf = ocr_result.get('confidence', 0)

                        if text and len(text) > 3:
                            successful_reads += 1
                            ocr_confs.append(ocr_conf)
                            results.append({
                                'image': img_path.name,
                                'text': text,
                                'det_conf': float(confidence),
                                'ocr_conf': ocr_conf
                            })
                            print(f"  [OK] {img_path.name}: {text} (Det:{confidence:.2f})")

            metrics = {
                'total_detections': total_detections,
                'successful_reads': successful_reads,
                'avg_det_time_ms': (total_time_det/len(self.test_images))*1000,
                'avg_ocr_time_ms': (total_time_ocr/total_detections)*1000 if total_detections > 0 else 0,
                'avg_det_conf': np.mean(det_confs) if det_confs else 0,
                'avg_ocr_conf': np.mean(ocr_confs) if ocr_confs else 0,
                'ocr_success_rate': (successful_reads/total_detections)*100 if total_detections > 0 else 0,
                'sample_results': results[:10]
            }

            self.results['OpenCV DNN (MobileNet-SSD)'] = metrics
            print(f"\nSummary: {successful_reads}/{total_detections} plates read successfully")
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_report(self):
        """Generate comparison report."""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "="*60)
        print("COMPARISON REPORT: NON-YOLO MODELS")
        print("="*60)

        # Sort by OCR success rate
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('ocr_success_rate', 0),
            reverse=True
        )

        print("\n1. OCR SUCCESS RATE (Most Important):")
        print("-"*60)
        for i, (name, metrics) in enumerate(sorted_results, 1):
            success = metrics.get('ocr_success_rate', 0)
            reads = metrics.get('successful_reads', 0)
            total = metrics.get('total_detections', 0)
            print(f"{i}. {name:30s} {success:5.1f}% ({reads}/{total} plates)")

        print("\n2. DETECTION SPEED:")
        print("-"*60)
        sorted_speed = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('avg_det_time_ms', float('inf'))
        )
        for i, (name, metrics) in enumerate(sorted_speed, 1):
            det_time = metrics.get('avg_det_time_ms', 0)
            print(f"{i}. {name:30s} {det_time:6.1f}ms")

        print("\n3. DETECTION CONFIDENCE:")
        print("-"*60)
        sorted_conf = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('avg_det_conf', 0),
            reverse=True
        )
        for i, (name, metrics) in enumerate(sorted_conf, 1):
            conf = metrics.get('avg_det_conf', 0)
            print(f"{i}. {name:30s} {conf:.3f}")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"non_yolo_comparison_{timestamp}.json"

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_images': len(self.test_images),
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved: {report_file}")
        print("="*60)


def main():
    """Main execution."""
    print("="*60)
    print("NON-YOLO DETECTION MODEL COMPARISON")
    print("Testing alternative detection frameworks")
    print("="*60)

    test_data_path = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test path not found: {test_data_path}")
        return

    comparison = NonYOLOComparison(test_data_path)

    # Test models
    print("\nTesting models...\n")
    comparison.test_custom_model()
    comparison.test_opencv_dnn()
    comparison.test_ssd_mobilenet()
    comparison.test_faster_rcnn()

    # Generate report
    comparison.generate_report()


if __name__ == "__main__":
    main()
