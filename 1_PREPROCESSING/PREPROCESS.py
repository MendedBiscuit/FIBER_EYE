import os
import cv2
import numpy as np

from pycocotools.coco import COCO


class Preprocessor:
    def __init__(self, A_pth, B_pth, G_pth, R_pth):
        self.channels = {
            "A": cv2.imread(A_pth),
            "B": cv2.imread(B_pth),
            "G": cv2.imread(G_pth),
            "R": cv2.imread(R_pth),
        }
        self.processed_dict = {}

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def normalise_intensity(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, woodchip_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        bg_mask = cv2.bitwise_not(woodchip_mask)

        light_map = cv2.GaussianBlur(bg_mask, (65, 65), 0)
        light_map[light_map < 1] = 1

        mean_bg_val = cv2.mean(gray, mask=bg_mask)[0]
        light_3ch = cv2.merge([light_map, light_map, light_map])
        corrected_img = cv2.divide(img, light_3ch, scale=mean_bg_val)

        woodchip_mask_3ch = cv2.merge([woodchip_mask, woodchip_mask, woodchip_mask])

        final_output = np.where(woodchip_mask_3ch == 255, img, corrected_img)

        return final_output.astype(np.uint8)

    def process_clahe_intensity(self):
        for key, img in self.channels.items():
            proc = self.normalise_intensity(img)
            self.processed_dict[key] = self.apply_clahe(proc)

        return self.processed_dict

    def CV_tile_and_save(self, sample_num, image_out, tile_size=512, stride=256):
        for key, img in self.processed_dict.items():
            h, w = img.shape[:2]
            tile_count = 0

            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    tile_image = img[y : y + tile_size, x : x + tile_size]

                    tile_id = f"{sample_num}_{key}_y{y}_x{x}"

                    cv2.imwrite(os.path.join(image_out, f"{tile_id}.png"), tile_image)
                    tile_count += 1

    def get_12_channel_stack(self):
        layers = [
            self.processed_dict["A"],
            self.processed_dict["B"],
            self.processed_dict["G"],
            self.processed_dict["R"],
        ]

        return np.concatenate(layers, axis=-1)

    def get_local_rms(self):
        img = self.processed_dict["A"].astype(np.float32) / 255.0

        mu = cv2.blur(img, (11, 11))
        mu_sq = cv2.blur(cv2.multiply(img, img), (11, 11))

        sigma = cv2.sqrt(cv2.absdiff(mu_sq, cv2.multiply(mu, mu)))
        return sigma

    def get_stack(self):
        rms = self.get_local_rms()
        all_channels = self.get_12_channel_stack()

        final_stack = np.concatenate([all_channels, rms], axis=-1)

        return final_stack

    def UNET_tile_and_save(
        self, sample_num, stack, mask, stack_out, mask_out, tile_size=512, stride=256
    ):
        h, w = stack.shape[:2]
        tile_count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile_stack = stack[y : y + tile_size, x : x + tile_size]
                tile_id = f"{sample_num}_y{y}_x{x}"

                np.savez_compressed(
                    os.path.join(stack_out, f"{tile_id}_multimodal.npz"),
                    image=tile_stack,
                )

                if mask is not None: 
                    tile_mask = mask[y : y + tile_size, x : x + tile_size]
                    cv2.imwrite(
                    os.path.join(mask_out, f"{tile_id}_multimodal.png"), tile_mask
                    )
                
                tile_count += 1


def masks_from_json(json_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coco = COCO(json_path)

    name_to_viz_id = {"wood_chip": 127, "impurity": 255}

    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        cat_lookup = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}

        for class_name in ["wood_chip", "impurity"]:
            original_cat_id = cat_lookup.get(class_name)
            if original_cat_id is None:
                continue

            viz_val = name_to_viz_id[class_name]
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[original_cat_id])
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                if isinstance(ann["segmentation"], list):
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], viz_val)
                else:
                    m = coco.annToMask(ann)
                    mask[m > 0] = viz_val

        raw_name = os.path.basename(img_info["file_name"])
        sample_num = raw_name.split("-")[-1].split("_")[0]
        target_name = f"{sample_num}_M.png"

        cv2.imwrite(os.path.join(output_dir, target_name), mask)
