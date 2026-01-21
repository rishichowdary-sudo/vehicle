"""
Script to pre-download PaddleOCR models.
Run this once on the server after deployment to ensure OCR works.

Usage:
    python download_ocr_models.py
"""

print("Downloading PaddleOCR models...")
print("This may take a few minutes on first run...")

try:
    from paddleocr import PaddleOCR
    
    # Initialize PaddleOCR - this triggers model download
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    print("\n✅ PaddleOCR models downloaded successfully!")
    print("Models are cached at: ~/.paddleocr/")
    
    # Quick test
    import numpy as np
    test_image = np.zeros((100, 300, 3), dtype=np.uint8)
    test_image.fill(255)  # White background
    
    result = ocr.ocr(test_image)
    print("✅ OCR test passed - ready for use!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure paddlepaddle and paddleocr are installed:")
    print("   pip install paddlepaddle paddleocr")
    print("2. Check internet connectivity for model download")
    print("3. Ensure you have write permissions to ~/.paddleocr/")
