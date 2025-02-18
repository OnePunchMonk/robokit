#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import os
import clip
import torch
import hydra
import logging
import warnings
import numpy as np
from PIL import Image as PILImg
import torchvision.transforms as tvT
from featup.util import norm, unnorm
from featup.plotting import plot_feats
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import predict
from groundingdino.util.utils import clean_state_dict
# from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from matplotlib import (patches, pyplot as plt)
import matplotlib.cm as cm


os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")

# resolve pyqt5 and cv2 issue
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class Logger(object):
    """
    This is a logger class.

    Attributes:
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Logger class.
        """
        super(Logger, self).__init__()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class Device(object):
    """
    This is a device class.

    Attributes:
        device (str): The device type ('cuda' or 'cpu').
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Device class.
        """
        super(Device, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class CommonContextObject(Logger, Device):
    """
    This is a common context object class.

    Attributes:
        logger: Logger instance for logging.
        device (str): The device type ('cuda' or 'cpu').
    """
    def __init__(self):
        """
        Initializes the CommonContextObject class.
        """
        super(CommonContextObject, self).__init__()


class FeatureUpSampler(CommonContextObject):
    """
    Root class for feature upsampling.
    All other feature upsampling classes should inherit this.

    Attributes:
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthPredictor class.
        """
        super(FeatureUpSampler, self).__init__()

    def upsample(self):
        """
        Upsample method for feature upscaling.
        Raises NotImplementedError as it should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        try:
            raise NotImplementedError("Upsample method must be implemented by subclasses")
        except NotImplementedError as e:
            self.logger.error(f"Error in upsample method: {e}")
            raise e


class DepthPredictor(CommonContextObject):
    """
    Root class for depth prediction.
    All other depth prediction classes should inherit this.

    Attributes:
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthPredictor class.
        """
        super(DepthPredictor, self).__init__()

    def predict(self):
        """
        Predict method for depth prediction.
        Raises NotImplementedError as it should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        try:
            raise NotImplementedError("predict method must be implemented by subclasses")
        except NotImplementedError as e:
            self.logger.error(f"Error in predict method: {e}")
            raise e


class FeatUp(FeatureUpSampler):
    """
    A class for upsampling features using a pre-trained backbone model.

    Attributes:
        input_size (int): Input size of the images.
        backbone_alias (str): Alias of the pre-trained backbone model.
        upsampler (torch.nn.Module): Feature upsampling module.
        logger (logging.Logger): Logger object for logging.
    """

    def __init__(self, backbone_alias, input_size, visualize_output=False):
        """
        Initializes the FeatUp class.

        Args:
            backbone_alias (str): Alias of the pre-trained backbone model.
            input_size (int): Input size of the images.
        """
        super(FeatUp, self).__init__()
        self.input_size = input_size
        self.backbone_alias = backbone_alias
        self.visualize_output = visualize_output
        self.img_transform = tvT.Compose([
            tvT.Resize(self.input_size),
            tvT.CenterCrop((self.input_size, self.input_size)),
            tvT.ToTensor(),
            norm
        ])
        try:
            self.upsampler = torch.hub.load("mhamilton723/FeatUp", self.backbone_alias).to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading FeatUp model: {e}")
            raise e

    def upsample(self, image_tensor):
        """
        Upsamples the features of encoded input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            Tuple: A tuple containing the original image tensor, backbone features, and upsampled features.
        """
        try:
            image_tensor = image_tensor.to(self.device)
            upsampled_features = self.upsampler(image_tensor) # upsampled features using backbone features; high resolution
            backbone_features = self.upsampler.model(image_tensor) # backbone features; low resolution
            orig_image = unnorm(image_tensor)
            batch_size = orig_image.shape[0]
            if self.visualize_output:
                self.logger.info("Plot input image with backbone and upsampled output")
                for i in range(batch_size):
                    plot_feats(orig_image[i], backbone_features[i], upsampled_features[i])
            return orig_image, backbone_features, upsampled_features

        except Exception as e:
            self.logger.error(f"Error during feature upsampling: {e}")
            raise e


class DepthAnythingPredictor(DepthPredictor):
    """
    A predictor class for depth estimation using a pre-trained model.

    Attributes:
        image_processor: Pre-trained image processor.
        model: Pre-trained depth estimation model.
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthAnythingPredictor class.
        """
        super(DepthPredictor, self).__init__() 
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.logger = logging.getLogger(__name__)

    def predict(self, img_pil):
        """
        Predicts depth from an input image.

        Args:
            PIL Image: Input image.

        Returns:
            PIL Image: Predicted depth map as a PIL image.
            numpy.ndarray: Predicted depth values as a numpy array.
        """
        try:
            image = img_pil.convert('RGB')

            # prepare image for the model
            inputs = self.image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # visualize the prediction
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_pil = PILImg.fromarray(formatted)
            return depth_pil, output

        except Exception as e:
            self.logger.error(f"Error predicting depth: {e}")
            raise e


class ObjectPredictor(CommonContextObject):
    """
    Root class for object predicton
    All other object prediction classes should inherit this
    """
    def __init__(self):
        super(ObjectPredictor, self).__init__()
    

    def bbox_to_scaled_xyxy(self, bboxes: torch.tensor, img_w, img_h):
        """
        Convert bounding boxes to scaled xyxy format.

        Parameters:
        - bboxes (torch.tensor): Input bounding boxes in cxcywh format.
        - img_w (int): Image width.
        - img_h (int): Image height.

        Returns:
        - torch.tensor: Converted bounding boxes in xyxy format.
        """
        try:
            bboxes = bboxes * torch.Tensor([img_w, img_h, img_w, img_h])
            bboxes_xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
            return bboxes_xyxy
        
        except Exception as e:
            self.logger.error(f"Error during bounding box conversion: {e}")
            raise e


class GroundingDINOObjectPredictor(ObjectPredictor):
    """
    This class implements Object detection using HuggingFace GroundingDINO
    Here instead of using generic language query, we fix the text prompt as "objects" which enables
    getting compact bounding boxes arounds generic objects.
    Hope is that these cropped bboxes when used with OpenAI CLIP yields good classification results.
    """
    def __init__(self):
        super(GroundingDINOObjectPredictor, self).__init__()
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
        self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"
        self.model = self.load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filenmae
        )
    

    def load_model_hf(self, model_config_path, repo_id, filename):
        """
        Load model from Hugging Face hub.

        Parameters:
        - model_config_path (str): Path to model configuration file.
        - repo_id (str): ID of the repository.
        - filename (str): Name of the file.

        Returns:
        - torch.nn.Module: Loaded model.
        """
        try:
            args = SLConfig.fromfile(model_config_path) 
            model = build_model(args)
            args.device = self.device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location=self.device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model    

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise e

    def image_transform_grounding(self, image_pil):
        """
        Apply image transformation for grounding.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - tuple: Tuple containing original PIL image and transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image_pil, image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for grounding: {e}")
            raise e

    def image_transform_for_vis(self, image_pil):
        """
        Apply image transformation for visualization.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - torch.tensor: Transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for visualization: {e}")
            raise e
    
    def predict(self, image_pil: PILImg, det_text_prompt: str = "objects"):
        """
        Get predictions for a given image using GroundingDINO model.
        Paper: https://arxiv.org/abs/2303.05499
        Parameters:
        - image_pil (PIL.Image): PIL.Image representing the input image.
        - det_text_prompt (str): Text prompt for object detection
        Returns:
        - bboxes (list): List of normalized bounding boxeS in cxcywh
        - phrases (list): List of detected phrases.
        - conf (list): List of confidences.

        Raises:
        - Exception: If an error occurs during model prediction.
        """
        try:
            _, image_tensor = self.image_transform_grounding(image_pil)
            bboxes, conf, phrases = predict(self.model, image_tensor, det_text_prompt, box_threshold=0.25, text_threshold=0.25, device=self.device)
            return bboxes, phrases, conf        
        except Exception as e:
            self.logger.error(f"Error during model prediction: {e}")
            raise e


class SegmentAnythingPredictor(ObjectPredictor):
    """
    Predictor class for segmenting objects using the Segment Anything model.

    Inherits from ObjectPredictor.

    Attributes:
    - device (str): The device used for inference, either "cuda" or "cpu".
    - sam (torch.nn.Module): The Segment Anything model.
    - mask_generator (SamAutomaticMaskGenerator): The mask generator for the SAM model.
    - predictor (SamPredictor): The predictor for the SAM model.
    """

    def __init__(self):
        """
        Initialize the SegmentAnythingPredictor object.
        """
        super(SegmentAnythingPredictor, self).__init__()
        self.sam = sam_model_registry["vit_t"](checkpoint="ckpts/mobilesam/vit_t.pth")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)  # generate masks for entire image
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    def predict(self, image, prompt_bboxes):
        """
        Predict segmentation masks for the input image.

        Parameters:
        - image: The input image as a numpy array.
        - prompt_bboxes: Optional prompt bounding boxes as a list of lists of integers [x_min, y_min, x_max, y_max].

        Returns:
        - A tuple containing the input bounding boxes (if provided) and the segmentation masks as torch Tensors.

        Raises:
        - ValueError: If the input image is not a numpy array.
        """
        try:
            # Convert input image to numpy array
            image = np.array(image)

            # Check if prompt_bboxes is provided
            if prompt_bboxes is not None:
                # Convert prompt bounding boxes to torch tensor
                input_boxes = torch.tensor(prompt_bboxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                self.predictor.set_image(image)
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                input_boxes = None
                masks = self.mask_generator.generate(image)
            
            return input_boxes, masks

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None, None


class ZeroShotClipPredictor(CommonContextObject):
    def __init__(self):
        super(ZeroShotClipPredictor, self).__init__()
        
        # Load the CLIP model
        self.model, self.preprocess = clip.load('ViT-L/14@336px', self.device)
        self.model.eval()

    def get_features(self, images, text_prompts):
        """
        Extract features from a list of images and text prompts.

        Parameters:
        - images (list of PIL.Image): A list of PIL.Image representing images.
        - text_prompts (list of str): List of text prompts.

        Returns:
        - Tuple of numpy.ndarray: Concatenated image features and text features as numpy arrays.

        Raises:
        - ValueError: If images is not a tensor or a list of tensors.
        - RuntimeError: If an error occurs during feature extraction.
        """
        try:

            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)
                _images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                img_features = self.model.encode_image(_images)
                text_features = self.model.encode_text(text_inputs)
            
            return img_features, text_features

        except ValueError as ve:
            self.logger.error(f"ValueError in get_image_features: {ve}")
            raise ve
        except RuntimeError as re:
            self.logger.error(f"RuntimeError in get_image_features: {re}")
            raise re

    def predict(self, image_array, text_prompts):
        """
        Run zero-shot prediction using CLIP model.

        Parameters:
        - image_array (List[torch.tensor]): List of tensor images.
        - text_prompts (list): List of text prompts for prediction.

        Returns:
        - Tuple: Tuple containing prediction confidence and indices.
        """
        try:
            # Perform prediction
            image_features, text_features = self.get_features(image_array, text_prompts)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pconf, indices = similarity.topk(1)

            return (pconf.flatten(), indices.flatten())

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error during prediction: {e}")
            raise e


class SAM2Predictor(ObjectPredictor):
    """
    Predictor class for video object segmentation using the SAM2 model.
    Source: https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/notebooks/video_predictor_example.ipynb
    """
    def __init__(self, text_prompt=None):
        """
        Initializes the SAM2Predictor class and attempts to load the model.
        """
        self.logger = logging.getLogger(__name__)        
        self.img_predictor, self.video_predictor = self._load_predictor()
        self.text_prompt = text_prompt


    def init_hydra_and_model_setup(self):
        # Source: https://github.com/facebookresearch/sam2/issues/81#issuecomment-2262979343
        # hydra is initialized on import of sam2, which sets the search path which can't be modified
        # so we need to clear the hydra instance
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        # reinit hydra with a new search path for configs
        hydra.initialize_config_module("robokit/sam2/sam2/", version_base='1.2') # Please don't change this

        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # Please don't change this
        self.checkpoint_path = "./ckpts/samv2/sam2.1_hiera_large.pth" # Please don't change this


    def _load_predictor(self):
        """
        Load the SAM2 model using the configuration and checkpoint path.
        """
        self.init_hydra_and_model_setup()
        img_predictor = self._load_img_predictor()
        video_predictor = self._load_video_predictor()
        return img_predictor, video_predictor


    def _load_img_predictor(self):
        """
        Load the SAM2 model using the configuration and checkpoint path for single image mask predictions.
        """
        try:
            # Load the SAM2 model with the configuration and checkpoint
            sam2_model = build_sam2(self.model_cfg, self.checkpoint_path)
            img_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 image predictor initialized successfully.")
            return img_predictor
        
        except Exception as e:
            print(f"Failed to load the predictor: {e}")
            raise

    def _load_video_predictor(self):
        """
        Load the SAM2 model using the configuration and checkpoint path for video mask predictions.
        """
        try:
            # Load the SAM2 model with the configuration and checkpoint
            predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path)
            print("SAM2 video predictor initialized successfully.")
            return predictor
        
        except Exception as e:
            print(f"Failed to load the predictor: {e}")
            raise

    def _load_and_show_frame(self, video_dir, frame_idx=0, show_frame=False):
        """
        Load and optionally display a specific video frame from a directory containing JPEG frames.

        Parameters:
        - video_dir: Directory containing video frames.
        - frame_idx: Index of the frame to display (default is 0).
        - show_frame: Boolean flag to indicate whether to display the frame (default is False).

        Returns:
        - img: The loaded image.
        """
        # Scan all image frames in the directory
        self.frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in self.get_valid_img_extensions()
        ]
        
        # Sort frames by the numeric part of the filename
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Load the specified frame
        if frame_idx < len(self.frame_names):
            frame_path = os.path.join(video_dir, self.frame_names[frame_idx])
            img = PILImg.open(frame_path)

            # Display the frame if requested
            if show_frame:
                plt.figure(figsize=(9, 6))
                plt.title(f"Frame {frame_idx}")
                plt.imshow(img)
                plt.axis('off')
                plt.show()

            return img
        else:
            print(f"Frame index {frame_idx} is out of range.")
            return None
    

    def rename_png_to_jpg(video_dir):
        """
        Rename all PNG images in the specified video directory to JPG,
        replacing '_color' with an empty string in the filenames.
        This is required as SAMv2 only reads mp4 and jpg/jpeg/JPG/JPEG files with frame names as 00000.jpg-....
        Parameters:
        - video_dir: Directory containing PNG images.
        """
        for filename in os.listdir(video_dir):
            if filename.endswith(".png"):
                # Replace '_color' with '' and change the file extension to .jpg
                new_filename = filename.replace('_color', '').replace('.png', '.jpg')
                old_path = os.path.join(video_dir, filename)
                new_path = os.path.join(video_dir, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

    def predict_mask_in_image(self, image_pil, prompt_bboxes):
        """
        Predict the mask for the given image using the provided bounding boxes.

        Args:
            image_pil (PIL.Image): The input image in PIL format.
            prompt_bboxes (np.array): [N,4] A list of bounding boxes to be used as the prompt for mask prediction.

        Returns:
            tuple: Contains the following elements:
                - masks (torch.Tensor): The predicted masks.
                - scores (torch.Tensor): The predicted scores.
                - logits (torch.Tensor): The predicted logits.
        """
        try:
            logging.info("Starting mask prediction.")

            # Convert image to numpy array
            image = np.array(image_pil.convert("RGB"))
            logging.debug("Image converted to numpy array.")

            # Set image for prediction
            self.img_predictor.set_image(image)
            logging.debug("Image set for predictor.")

            # Predict masks, scores, and logits
            masks, scores, logits = self.img_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=prompt_bboxes,
                multimask_output=False,
            )
            logging.info("Mask prediction completed.")

            return masks, scores, logits

        except Exception as e:
            logging.error(f"An error occurred during mask prediction: {e}")
            raise  # Re-raise the exception to propagate it up


    def propagate_point_prompt_masks_and_save(self, video_dir, point_prompts, save_output=True):
        """
        Propagate the segmentation mask across the entire video and optionally save the frames with masks to a subdirectory.
        Assumption: Only one object in the frame + only one box prompt is given as prompt
        Parameters:
        - video_dir: Path to the video frames directory.
        - point_prompts: List of object point prompts as [[(x1,y1, pos-1), [x2,y2,neg-0], ...]]
            -  each element is a list of point prompts for an object
                -  x,y is the pixel location
                -  pos is indicates positive label
                -  neg indicated negative label
                -  x and y are needed for point params
                -  pos and neg are required for label params
        - save_output: If True, saves the segmented frames (default is True).
        """
        try:
            # Initialize inference state
            inference_state = self.video_predictor.init_state(video_path=video_dir)
            self.video_predictor.reset_state(inference_state)
            
            # Get all frames from the directory
            frame_names = self.load_frames_from_directory(video_dir)
            
            prompts = {}  # hold all the clicks we add for visualization

            # Segment first frame
            for obj_idx, obj_point_prompts in enumerate(point_prompts):
                frame_idx = 0

                points, labels = (
                    np.array([[x, y] for x, y, _ in obj_point_prompts], dtype=np.float32),
                    np.array([label for _, _, label in obj_point_prompts], dtype=np.int32)
                )

                prompts[obj_idx] = points, labels

                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_idx,
                    points=points,
                    labels=labels,
                )

            # Create output directory if save_output is True
            if save_output:
                out_path_suffix = "objects"
                output_dir = os.path.join(os.path.dirname(video_dir), f"out/samv2/{out_path_suffix}")
                masks_dir = os.path.join(output_dir, "masks")
                os.makedirs(masks_dir, exist_ok=True)

            # Propagate mask across video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()}
                    # logging.info("Visualize and optionally save the results")
                    plt.close("all")
            
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Frame {out_frame_idx}")
                    img_path = os.path.join(video_dir, frame_names[out_frame_idx])
                    img = PILImg.open(img_path)
                    plt.imshow(img)
                    
                    out_file_name = frame_names[out_frame_idx].replace('.jpg', '.png')

                    # Show segmentation masks
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():

                        for i, out_obj_id in enumerate(out_obj_ids):
                            self.show_points(*prompts[out_obj_id], plt.gca())
                            self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                        # self.show_mask(out_mask[0], plt.gca(), obj_id=out_obj_id)
                        
                        # Turn off axis labels
                        plt.axis('off')
                        
                        # Save the mask overlayed image result if save_output is True
                        if save_output:
                            plt.savefig(os.path.join(masks_dir, out_file_name))
            
        except Exception as e:
            logging.error(f"Error during mask propagation: {e}")

        return frame_names, video_segments

    def propagate_masks_and_save(self, video_dir, bboxes, save_output=True):
        """
        Propagate the segmentation mask across the entire video and optionally save the frames with masks to a subdirectory.
        Assumption: Only one object in the frame + only one box prompt is given as prompt
        Parameters:
        - video_dir: Path to the video frames directory.
        - bboxes: The List of bounding boxes [[x_min, y_min, x_max, y_max]] for initial segmentation.
        - save_output: If True, saves the segmented frames (default is True).
        """
        try:
            # Initialize inference state
            inference_state = self.video_predictor.init_state(video_path=video_dir)
            self.video_predictor.reset_state(inference_state)
            
            # Get all frames from the directory
            frame_names = self.load_frames_from_directory(video_dir)

            # Segment first frame
            for obj_idx, obj_bbox in enumerate(bboxes):
                frame_idx = 0
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_idx,
                    box=obj_bbox
                )
            


            # Create output directory if save_output is True
            if save_output:
                out_path_suffix = f"/{self.text_prompt.lower().replace(' ', '_')}" if self.text_prompt else ''
                output_dir = os.path.join(os.path.dirname(video_dir), f"out/samv2{out_path_suffix}")
                masks_dir = os.path.join(output_dir, "masks")
                traj_overlayed_dir = os.path.join(output_dir, "traj_overlayed")
                os.makedirs(masks_dir, exist_ok=True)
                os.makedirs(traj_overlayed_dir, exist_ok=True)
            
            # Propagate mask across video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()}
                    # logging.info("Visualize and optionally save the results")
                    plt.close("all")
            
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Frame {out_frame_idx}")
                    img_path = os.path.join(video_dir, frame_names[out_frame_idx])
                    img = PILImg.open(img_path)
                    plt.imshow(img)
                    
                    out_file_name = frame_names[out_frame_idx].replace('.jpg', '.png')

                    # Show segmentation masks
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        for i, out_obj_id in enumerate(out_obj_ids):
                            self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                        # self.show_mask(out_mask[0], plt.gca(), obj_id=out_obj_id)
                        # Turn off axis labels
                        plt.axis('off')
                        
                        # Save the mask overlayed image result if save_output is True
                        if save_output:
                            plt.savefig(os.path.join(masks_dir, out_file_name))
                                    
        except Exception as e:
            logging.error(f"Error during mask propagation: {e}")

        return frame_names, video_segments

    def calculate_centroid(self, mask):
        """
        Calculate the centroid of the object in the mask.
        """
        # Assuming the mask is a binary numpy array with the object pixels as 1
        _, y_indices, x_indices = np.where(mask == 1)
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        return (centroid_x, centroid_y)

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        """
        Displays the segmentation mask on the given axes.
        
        Parameters:
        - mask: The segmentation mask.
        - ax: The axes on which to display the mask.
        - obj_id: Optional object ID to color code the mask.
        - random_color: If True, assigns a random color to the mask.
        """
        try:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                color = np.array([*cmap(obj_id or 0)[:3], 0.6])

            mask_image = mask.reshape(mask.shape[-2], mask.shape[-1], 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        except ValueError as e:
            self.logger.error(f"Error in show_mask: {e}")
            raise

    def show_points(self, coords, labels, ax, marker_size=200):
        """
        Displays points on the image based on their labels.
        
        Parameters:
        - coords: Coordinates of the points.
        - labels: Labels for the points (e.g., positive or negative).
        - ax: The axes for displaying the points.
        - marker_size: The size of the point markers.
        """
        try:
            pos_points = coords[labels == 1]
            neg_points = coords[labels == 0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        except IndexError as e:
            self.logger.error(f"Error in show_points: {e}")
            raise

    def show_box(self, box, ax):
        """
        Draws a bounding box on the axes.
        
        Parameters:
        - box: Bounding box in the format [x_min, y_min, x_max, y_max].
        - ax: The axes to draw the bounding box on.
        """
        try:
            if len(box) != 4:
                raise ValueError("Box must contain exactly 4 values: [x_min, y_min, x_max, y_max].")
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        except ValueError as e:
            self.logger.error(f"Error in show_box: {e}")
            raise

    def get_valid_img_extensions(self):
        """
        Returns a list of valid image file extensions.
        """
        return [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    def load_frames_from_directory(self, video_dir):
        """
        Loads frame names from the directory that are valid image files.

        Parameters:
        - video_dir: Directory containing video frames.

        Returns:
        - List of valid image file names.
        """
        valid_extensions = self.get_valid_img_extensions()
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in valid_extensions
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names

    def create_collage(self, video_dir, collage_size=(2, 3)):
        """
        Creates a collage of video frames for visualization.
        
        Parameters:
        - video_dir: Directory of video frames.
        - collage_size: The number of rows and columns in the collage.
        """
        try:
            frame_names = self.load_frames_from_directory(video_dir)
            fig, axes = plt.subplots(collage_size[0], collage_size[1], figsize=(12, 8))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < len(frame_names):
                    img = PILImg.open(os.path.join(video_dir, frame_names[i]))
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.logger.error(f"Error creating collage: {e}")
            raise
