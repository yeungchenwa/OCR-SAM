import cv2
import numpy as np

mask_img_path = "G:/Code/DLVC/opensource_repo/OCR-SAM/results/erase_4_sam/whole_mask.jpg"

mask_rgb_img = cv2.imread(mask_img_path)
mask_img = cv2.cvtColor(mask_rgb_img, cv2.COLOR_RGB2GRAY)

kernel = np.ones((3, 3), np.int8)
dilated_mask_img = cv2.dilate(mask_img, kernel, iterations=1)
cv2.imwrite(f"G:/Code/DLVC/opensource_repo/OCR-SAM/results/erase_4_sam/dilated_mask_3x3.jpg", dilated_mask_img)
