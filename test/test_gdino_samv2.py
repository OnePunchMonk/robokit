#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

"""
This script performs object tracking on video frames using Grounding DINO for initial detection 
and SAM2 for tracking each detected bounding box across frames.

The script follows these steps:

1. **Initialize Flags**:
   - Define command-line flags for the input directory containing video frames, a text prompt for object detection,
     and an interval for saving frames during tracking.

2. **Main Function**:
   - Parse the command-line flags to obtain the input directory, text prompt, and save interval.
   - Ensure both input directory and text prompt are provided; otherwise, raise an error.

3. **Setup Object Detectors**:
   - Initialize the Grounding DINO object detector to detect bounding boxes in the first frame based on the given text prompt.
   - Initialize SAM2 to track detected bounding boxes across all frames.

4. **Process First Frame**:
   - Load the first frame from the input directory.
   - Use Grounding DINO to detect initial bounding boxes with the specified text prompt.

5. **Track All Detected Bounding Boxes**:
   - For each bounding box detected in the first frame:
     a. Convert the bounding box to a [x_min, y_min, x_max, y_max] format.
     b. Pass each bounding box to SAM2 to propagate the bounding box across all frames, tracking the object's movement.
     c. Set the tracking save interval to the specified value for periodic saving of the tracking results.

6. **Output**:
   - Each bounding box's tracking results are saved in the specified save interval.
   - Log messages indicate the start and completion of tracking for each bounding box.

**Usage**:
   - Run this script from the command line, passing the required `input_dir` and `text_prompt` flags, 
     and an optional `save_interval` (default is 1):
   
     `python test_gdino_samv2.py --input_dir=<path_to_frames> --text_prompt="object description" --save_interval=2`
"""


import os
import numpy as np
from PIL import Image as PILImg
from absl import app, flags, logging
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SAM2Predictor

# Define absl flags for CLI arguments
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None, 'Directory path to input video frames')
flags.DEFINE_string('text_prompt', None, 'Text prompt for initial object detection')
flags.DEFINE_integer('save_interval', 1, 'Interval for saving tracked frames')


def main(argv):
    # Get input values from flags
    video_dir, text_prompt = sanity_check(argv)
    save_interval = FLAGS.save_interval

    # Check if required flags are provided
    if not video_dir or not text_prompt:
        raise ValueError("Both --input_dir and --text_prompt flags must be provided.")

    logging.info("Initialize object detectors")

    # Initialize Grounding DINO for initial bbox detection
    gdino = GroundingDINOObjectPredictor()

    # Initialize SAM2 for tracking across frames
    sam2 = SAM2Predictor(text_prompt)

    for img_path in sorted(os.listdir(video_dir)):
        img_name = img_path
        img_path = os.path.join(video_dir, img_path)
        try:
            # Read the first frame to detect initial bounding boxes
            img_pil = PILImg.open(img_path).convert("RGB")

            logging.info("GDINO: Predict initial bounding boxes, phrases, and confidence scores")
            initial_bboxes, phrases, gdino_conf = gdino.predict(img_pil, text_prompt)

            # Scale bounding boxes to match the original image 
            w, h = img_pil.size # Get image width and height 
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(initial_bboxes, w, h)
            
            logging.info(f"Run SAM2 using bounding box on current frame")
            masks, scores, logits = sam2.predict_mask_in_image(img_pil, image_pil_bboxes)
            
            logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
            # bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)

            # import pdb; pdb.set_trace()
            
            # bbox_annotated_pil.show()

            print(masks.shape, scores)

            result_masks = []

            index = np.argmax(scores)
            result_masks.append(masks[index])
            #print(type(result_masks)) #Animesh Update
            masks = np.array(result_masks)

            image_array = np.array(masks, dtype=np.uint8) #Animesh Update
        
            img = np.zeros((480, 640), dtype=np.uint8)
            print(img.shape)
            box_size = dict()
            for p in range(image_array.shape[0]-1, -1, -1):
                one_count = 0
                one_count = np.sum(image_array[p] == 1)
                box_size[p] = one_count
            val = max(box_size, key=box_size.get)   
            background_mask = val
            print(img.shape)
            for p in range(image_array.shape[0]-1, -1, -1):
                if p==background_mask:
                    continue
                img = np.where(image_array[p] == 1, p+1, img)

            # background_mask = -1
            # for p in range(image_array.shape[0]-1, -1, -1):
            #     one_count = 0
            #     one_count = np.sum(image_array[p] == 1)
                        
            #     if (one_count/307200)<=0.5:
            #         img = np.where(image_array[p] == 1, assign, img)
            #         assign+=1
            print(img.shape)
            to_save = PILImg.fromarray(img)
            to_save.save('./gsam2_masks/'+img_name)
            print("Completed Processing: " + img_name)

        except Exception as e:
            # Handle unexpected errors
            print(f"An unexpected error occurred: {e}")
        
        # break


def sanity_check(argv):
    input_dir = flags.FLAGS.input_dir
    text_prompt = flags.FLAGS.text_prompt
    
    # Check if the input directory exists and is a valid directory
    if not os.path.isdir(input_dir):
        raise Exception(f"Error: The directory '{input_dir}' does not exist or is not a valid directory.")
        

    # List all files in the directory and filter for image files (jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise Exception(f"No image files found in the directory '{input_dir}'.")
    
    # Ensure text_prompt is provided
    if not text_prompt:
        raise Exception("Error: 'text_prompt' is required but not provided.")

    return input_dir, text_prompt


if __name__ == "__main__":
    # Mark flags as required
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('text_prompt')
    
    # Run the main function
    app.run(main)
