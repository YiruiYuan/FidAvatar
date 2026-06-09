import os
import cv2
import numpy as np

img_dir = "./data/insta/yyr/images"
mask_dir = "./data/insta/yyr/alpha"

files = sorted(os.listdir(img_dir))
for f in files:
    if not f.endswith(".png"): continue
    
    img_path = os.path.join(img_dir, f)
    mask_path = os.path.join(mask_dir, f)
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue
        
    if img.shape[-1] == 3:
        # Convert RGB to RGBA by appending the mask as the alpha channel
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        cv2.imwrite(img_path, rgba)

print("Completed: images are RGBA")
