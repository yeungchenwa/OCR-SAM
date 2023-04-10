import cv2
import PIL.Image as Image
import torch
# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
# SAM
from segment_anything import SamPredictor, sam_model_registry

from diffusers import StableDiffusionInpaintPipeline

if __name__ == '__main__':
    det_config = 'mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py'
    det_weight = 'mmocr_dev/checkpoints/db_swin_mix_pretrain.pth'
    rec_config = 'mmocr_dev/configs/textrecog/unirec/unirec.py'
    rec_weight = 'mmocr_dev/checkpoints/unirec.pth'
    sam_checkpoint = 'segment-anything-main/checkpoints/sam_vit_h_4b8939.pth'
    prompt = 'Text like a cake'
    sam_type = 'vit_h'
    device = 'cuda'
    img_path = 'example_images/ex2.jpg'
    select_index = 0

    # MMOCR
    mmocr_inferencer = MMOCRInferencer(
        det_config, det_weight, rec_config, rec_weight, device=device)
    # SAM
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    # Diffuser
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    img = cv2.imread(img_path)
    result = mmocr_inferencer(img)['predictions'][0]
    rec_texts = result['rec_texts']
    det_polygons = result['det_polygons']
    det_bboxes = torch.tensor([poly2bbox(poly) for poly in det_polygons],
                              device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        det_bboxes, img.shape[:2])
    # SAM inference
    sam_predictor.set_image(img, image_format='BGR')
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # Diffuser inference
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    mask = masks[select_index][0].cpu().numpy()
    mask = Image.fromarray(mask)
    image = pipe(
        prompt=prompt,
        image=img.resize((512, 512)),
        mask_image=mask.resize((512, 512))).images[0]
    image.save('test_out.png')