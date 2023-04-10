import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, img_size, device):
    if img_size is not None:
        image = np.array(Image.open(image).convert("RGB").resize(img_size))
    else:
        image = np.array(Image.open(image).convert("RGB")) # need to resize to a image_size
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    if img_size is not None:
        mask = np.array(Image.open(mask).convert("L").resize(img_size))
    else:
        mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def erase_text(opt,):
    # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    # masks = [m.replace('\\', '/') for m in masks]
    # images = [x.replace("_mask.png", ".png") for x in masks]
    masks = [opt.mask_image]
    images = [opt.image]
    # print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("checkpoints/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.out_dir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                batch = make_batch(image, 
                                   mask, 
                                   img_size=opt.img_size,
                                   device=device)

                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(f"{opt.out_dir}/erased_image.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        nargs="?",
        default="G:/Code/DLVC/opensource_repo/OCR-SAM/results/ldm_inpaint_input_3/ex3.png",
        help="The image will be erased",
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        nargs="?",
        default="G:/Code/DLVC/opensource_repo/OCR-SAM/results/ldm_inpaint_input_3/ex3_mask.png",
        help="The mask image will be the prompt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        default="G:/Code/DLVC/opensource_repo/OCR-SAM/results/ldm_inpaint_output_3",
        help="dir to write results to",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    erase_text(opt=opt)