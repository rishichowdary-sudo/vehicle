"""
Image preprocessing utilities for better plate detection
"""

import cv2
import numpy as np


def enhance_image(image):
    """Enhance image quality for better detection.

    Args:
        image: Input image (numpy array)

    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels
    enhanced = cv2.merge([l, a, b])

    # Convert back to BGR
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def denoise_image(image):
    """Remove noise from image.

    Args:
        image: Input image

    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def sharpen_image(image):
    """Sharpen image to enhance edges.

    Args:
        image: Input image

    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])

    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def preprocess_for_detection(image_path):
    """Preprocess image for optimal plate detection.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Enhance contrast
    enhanced = enhance_image(image)

    # Sharpen
    sharpened = sharpen_image(enhanced)

    return sharpened


def resize_image(image, max_width=1920, max_height=1080):
    """Resize image to fit within maximum dimensions.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if w <= max_width and h <= max_height:
        return image

    # Calculate scaling factor
    scale = min(max_width / w, max_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized


def extract_plate_region(image, bbox, padding=10):
    """Extract plate region with padding.

    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding around the plate

    Returns:
        Cropped plate image
    """
    x1, y1, x2, y2 = bbox

    # Add padding
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Extract region
    plate = image[y1:y2, x1:x2]

    return plate


def correct_skew(image):
    """Correct skew/rotation in plate image.

    Args:
        image: Plate image

    Returns:
        Deskewed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            angles.append(angle)

        median_angle = np.median(angles)

        # Rotate image
        if abs(median_angle - 90) > 5:  # If significantly skewed
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_angle = median_angle - 90

            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated

    return image
