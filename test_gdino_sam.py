# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

import os
import torch
from pathlib import Path
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor


def main(argv):
    # Path to the input image 
    root_dir = Path("/home/jishnu/Projects/rdd/assets/gazebo-cracker_box/rgb")
    mask_out_dir = root_dir / "../masks"
    os.makedirs(mask_out_dir, exist_ok=True)

    for i in os.listdir(root_dir):
        image_path = root_dir / i
        text_prompt =  'redbox'

        try:
            logging.info("Initialize object detectors")
            gdino = GroundingDINOObjectPredictor()
            SAM = SegmentAnythingPredictor()

            logging.info("Open the image and convert to RGB format")
            image_pil = PILImg.open(image_path).convert("RGB")
            
            logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

            logging.info("GDINO post processing")
            w, h = image_pil.size # Get image width and height 
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            logging.info("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

            # bbox_annotated_pil.show()

            # Assume `masks` is your tensor: shape [1, 1, 480, 640] on CUDA
            mask_cpu = masks.squeeze().to(torch.uint8).cpu() * 255  # shape: [480, 640], values: 0 or 255

            # Convert to PIL image
            mask_img = PILImg.fromarray(mask_cpu.numpy())

            # Save the mask image
            mask_img_path = mask_out_dir / i
            mask_img.save(mask_img_path)

        except Exception as e:
            # Handle unexpected errors
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    # app.run(main, ['imgs/irvl-clutter-test.png'])
    app.run(main, ['imgs/irvl-clutter-test-2.png'])