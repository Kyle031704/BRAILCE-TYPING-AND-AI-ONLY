import cv2
import numpy as np

img_path = r'C:\Users\vangie\.gemini\antigravity-ide\brain\9594dcf3-f045-45a4-ab12-d453eaf4a9fd\.user_uploaded\media_1792937268123.png'
img = cv2.imread(img_path)

def enhance_braille(image):
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

enhanced = enhance_braille(img)
cv2.imwrite("tools/enhanced_braille.jpg", enhanced)
print("Enhanced image saved!")
