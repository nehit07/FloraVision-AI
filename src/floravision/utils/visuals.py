"""
FloraVision AI - Visual Utilities
================================

PURPOSE:
    Provides helper functions for image manipulation and data visualization.
"""

from PIL import Image, ImageDraw, ImageFont
import io
from typing import List
from ..state import YOLODetection

def draw_detections(image_bytes: bytes, detections: List[YOLODetection]) -> bytes:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image_bytes: Original image bytes
        detections: List of YOLODetection objects with box coordinates
        
    Returns:
        Annotated image bytes
    """
    if not detections or not any(d.box for d in detections):
        return image_bytes
        
    try:
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Use a high-contrast color palette for symptoms
        colors = ["#FF3838", "#FF9D00", "#FFD700", "#2ECC71", "#3498DB", "#9B59B6"]
        
        for i, det in enumerate(detections):
            if not det.box:
                continue
                
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = det.box
            
            # Convert normalized coordinates to pixel coordinates
            # Note: YOLO coordinates are [x1, y1, x2, y2]
            left = x1 * width
            top = y1 * height
            right = x2 * width
            bottom = y2 * height
            
            # Draw box
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            # Draw label background
            label = f"{det.label.replace('_', ' ').title()} {int(det.confidence * 100)}%"
            
            # Simple fallback for font
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
                
            # Draw text
            draw.text((left + 5, top + 5), label, fill=color, font=font)
            
        # Save back to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()
        
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return image_bytes
