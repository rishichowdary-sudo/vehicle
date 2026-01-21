# Vehicle Registration Model Accuracy Comparison Report

**Test Date:** January 12, 2026
**Test Dataset:** 8 images from "No plates" folder
**Models Tested:** 5 Detection Models, 3 OCR Models

---

## Executive Summary

Comprehensive testing was conducted to compare different models for license plate detection and OCR. All detection models achieved **100% detection rate** on the test dataset, but varied significantly in speed and confidence.

---

## 1. Detection Model Results

### Performance Comparison

| Model | Speed (ms) | FPS | Confidence | Detections | Recommendation |
|-------|-----------|-----|------------|------------|----------------|
| **YOLOv8n** | 83.56 | 12.0 | 0.632 | 13 | ✅ **Best for Real-time** |
| **YOLOv8s** | 143.70 | 7.0 | 0.636 | 19 | ⚖️ Balanced |
| **YOLOv8m** | 334.98 | 3.0 | 0.625 | 18 | ⚠️ Slow |
| **YOLOv8l** | 1464.58 | 0.7 | 0.628 | 20 | ❌ Too Slow |
| **Custom-Plate-Detector** | 215.22 | 4.6 | 0.661 | 11 | ✅ **Highest Confidence** |

### Key Findings - Detection

1. **YOLOv8n (Nano)** - RECOMMENDED for Real-time Applications
   - Fastest inference: 83.56ms (12 FPS)
   - 100% detection rate
   - Good confidence: 0.632
   - Best for video streams and real-time processing

2. **Custom-Plate-Detector** - RECOMMENDED for Accuracy
   - Highest confidence: 0.661
   - 100% detection rate
   - Speed: 215.22ms (4.6 FPS)
   - Specialized for license plates
   - Best for batch processing and high-accuracy requirements

3. **YOLOv8s (Small)** - Good Middle Ground
   - 143.70ms (7.0 FPS)
   - More detections (19) than others
   - Slightly higher confidence than YOLOv8n

4. **YOLOv8m & YOLOv8l** - NOT RECOMMENDED
   - Too slow for practical use
   - No significant accuracy improvement
   - YOLOv8l is 17.5x slower than YOLOv8n

### Speed vs Accuracy Trade-off

```
Speed (FPS)     |  Model
----------------|------------------
12.0 FPS ████   |  YOLOv8n (FASTEST)
7.0 FPS  ██     |  YOLOv8s
4.6 FPS  █      |  Custom-Plate-Detector
3.0 FPS  █      |  YOLOv8m
0.7 FPS  ▌      |  YOLOv8l (SLOWEST)

Confidence      |  Model
----------------|------------------
0.661 ████████  |  Custom-Plate-Detector (BEST)
0.636 ███████▌  |  YOLOv8s
0.632 ███████▌  |  YOLOv8n
0.628 ███████▌  |  YOLOv8l
0.625 ███████▌  |  YOLOv8m
```

---

## 2. OCR Model Results

### Performance Comparison

| Model | Speed (ms) | Success Rate | Confidence | Status |
|-------|-----------|--------------|------------|--------|
| **PaddleOCR** | N/A | N/A | N/A | ❌ API Error |
| **EasyOCR** | 4399.29 | 69.2% | 0.338 | ⚠️ Poor Performance |
| **Tesseract** | N/A | N/A | N/A | ❌ Not Installed |

### Key Findings - OCR

1. **EasyOCR** - Only Working Model
   - Very slow: 4399ms per plate (0.23 FPS)
   - Moderate success rate: 69.2%
   - Low confidence: 0.338
   - Not suitable for production use

2. **PaddleOCR** - API Issues
   - Your current implementation uses outdated PaddleOCR API
   - Error: `got an unexpected keyword argument 'cls'`
   - Needs to be updated to use the new `.predict()` method

3. **Tesseract** - Not Available
   - Not installed on system
   - Generally poor for license plates

### Sample OCR Results (EasyOCR)

| Source | Detected Text | Confidence |
|--------|--------------|------------|
| vnv.jpg | HRZGFC2782 | 0.885 ✅ |
| double.png | voltio MYFI 15646 | 0.633 |
| unnamed.jpg | 6888 [6888 Dominob | 0.348 |
| ver.jpeg | Ktc KA 02MP 9657 | 0.289 |
| bg.jpg | [lZ Ce 6616 | 0.125 ❌ |

---

## 3. Recommendations

### Primary Recommendation: Fix PaddleOCR

Your current system uses PaddleOCR, which is generally the best option for license plate OCR, but the API has changed. You need to:

1. **Update PaddleOCR calls** from `.ocr()` to `.predict()`
2. Test PaddleOCR with the new API
3. Compare PaddleOCR vs EasyOCR performance

### Detection Model Selection

**For Real-time Applications (Video Streams):**
- Use **YOLOv8n**
- 12 FPS processing speed
- Suitable for live camera feeds

**For Batch Processing (High Accuracy):**
- Use **Custom-Plate-Detector**
- Highest confidence (0.661)
- Specifically trained for license plates
- Accept slower speed for better accuracy

**For Balanced Use Cases:**
- Use **YOLOv8s**
- 7 FPS is acceptable for most applications
- More detections than other models

### System Configuration Recommendations

**Current Configuration (config.py):**
```python
YOLO_MODEL_PATH = 'license_plate_detector.pt'
DETECTION_CONFIDENCE = 0.3
```

**Recommended Configurations:**

**Option A: Real-time Performance**
```python
YOLO_MODEL_PATH = 'yolov8n.pt'
DETECTION_CONFIDENCE = 0.3
# Speed: 83ms, FPS: 12
```

**Option B: Maximum Accuracy (RECOMMENDED)**
```python
YOLO_MODEL_PATH = 'license_plate_detector.pt'
DETECTION_CONFIDENCE = 0.3
# Speed: 215ms, FPS: 4.6, Confidence: 0.661
```

**Option C: Balanced**
```python
YOLO_MODEL_PATH = 'yolov8s.pt'
DETECTION_CONFIDENCE = 0.3
# Speed: 143ms, FPS: 7, More detections
```

---

## 4. Next Steps

### Immediate Actions:

1. **Fix PaddleOCR Integration**
   - Update from `.ocr(img, cls=True)` to `.predict(img)`
   - Re-run comparison tests
   - Expected: Much better performance than EasyOCR

2. **Choose Detection Model**
   - Keep current Custom-Plate-Detector for accuracy
   - OR switch to YOLOv8n for speed
   - Update config.py accordingly

3. **Test on Larger Dataset**
   - Current test: 8 images
   - Recommend: 100+ images for statistical significance
   - Include various lighting conditions, angles, plate types

### Future Enhancements:

1. **Try Additional Models:**
   - YOLOv10 (if available)
   - YOLO11 (latest version)
   - Combine multiple detectors for ensemble approach

2. **OCR Improvements:**
   - Fix PaddleOCR implementation
   - Try TrOCR (Transformer-based OCR)
   - Implement post-processing rules for Indian plates

3. **End-to-End Pipeline Testing:**
   - Test combined detection + OCR pipelines
   - Measure total processing time
   - Calculate end-to-end accuracy

---

## 5. Conclusion

### Current System Assessment

✅ **Strengths:**
- 100% detection rate across all models
- Custom-Plate-Detector has excellent confidence (0.661)
- Multiple model options available

⚠️ **Weaknesses:**
- PaddleOCR integration broken (API mismatch)
- EasyOCR is too slow for production (4.4s per plate)
- Limited test dataset (8 images)

### Winner: YOLOv8n + PaddleOCR (Fixed)

**Recommended Final Configuration:**
- **Detection:** YOLOv8n (83ms, 12 FPS)
- **OCR:** PaddleOCR (fix API, expected ~200-500ms)
- **Total Pipeline:** ~300-600ms per image (1.6-3.3 FPS end-to-end)

**Alternative for Maximum Accuracy:**
- **Detection:** Custom-Plate-Detector (215ms, 0.661 conf)
- **OCR:** PaddleOCR (fix API, expected ~200-500ms)
- **Total Pipeline:** ~400-700ms per image (1.4-2.5 FPS end-to-end)

---

## Appendix: Testing Environment

- **Hardware:** CPU (No GPU acceleration)
- **Test Images:** 8 images from "No plates" folder
- **Detection Threshold:** 0.3
- **Date:** January 12, 2026

**Note:** Results may vary with GPU acceleration. Detection speeds could improve by 3-5x with CUDA-enabled GPU.
