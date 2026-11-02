import sys
sys.path.insert(0, ".")
import cv2
from brailce.vision.obr_engine import OBREngine

img_path = r'C:\Users\vangie\.gemini\antigravity-ide\brain\9594dcf3-f045-45a4-ab12-d453eaf4a9fd\.user_uploaded\media_1792937268123.png'
img = cv2.imread(img_path)

for conf in [0.25, 0.22, 0.18, 0.15, 0.12, 0.08]:
    engine = OBREngine(model_path='brailce/vision/yolov8_braille.pt', conf=conf)
    res = engine.process_image(img)
    readable = res.get("readable")
    txt = res.get("text", "")
    print(f"conf={conf:.2f} -> readable={readable}, length={len(txt)}, sample={repr(txt[:60])}")
