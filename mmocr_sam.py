from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import register_all_modules
register_all_modules()
inputs = 'segment-anything-main/images/img1.jpg'
out_dir = 'results/'
det = 'FCENet'
rec = 'ABINet'
ocr = MMOCRInferencer(det=det, rec=rec, device='cuda')
ocr(inputs, out_dir=out_dir, print_result=True, save_vis=True)