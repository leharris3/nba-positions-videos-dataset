from easy_ViTPose.vit_utils.inference import pad_image
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def pre_process_image(img, model):

    # TODO: we are gettting occasional ZeroDivisionError errors
    # using a pretty shitty workaround atm
    try:
        img, _ = pad_image(img, 3 / 4)
        img_input, org_h, org_w = model.pre_img(img)
    except Exception as e:
        logger.error(f"Failed to pad image with shape {img.shape}, error: {e}")
        # use a dummy image
        img = np.zeros((256, 192, 3), dtype=np.uint8)
        img, _ = pad_image(img, 3 / 4)
        img_input, org_h, org_w = model.pre_img(img)
    return img_input, (org_h, org_w)


def post_process_image(i, org_w, org_h, heatmaps, model):
    heatmap = np.expand_dims(heatmaps[i, :, :, :], axis=0)
    return model.postprocess(heatmap, org_w, org_h)[0]
