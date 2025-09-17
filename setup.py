#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import os
import sys
import subprocess
import setuptools
import logging
from setuptools.command.install import install


class FileFetch(install):
    """
    Custom setuptools command to fetch required files from external sources.
    """
    def run(self):
        """
        Execute the command to fetch required files.
        """
        install.run(self)

        rkit_root_dir = os.getcwd()


        # Install the dependency from the Git repository
        subprocess.run([
            "pip", "install", "-U",
            'git+https://github.com/mhamilton723/FeatUp@c04e4c19945ce3e98a5488be948c7cc1fdcdacc6',
            'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
            'git+https://github.com/IDEA-Research/GroundingDINO.git@2b62f419c292ca9c518daae55512fabc3fead4a4',
            'git+https://github.com/ChaoningZhang/MobileSAM@c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed',
        ])

        # Step DHYOLO.1: Clone the DH-YOLO repository
        try:
            subprocess.run(["git", "clone", "https://github.com/IRVLUTD/iTeach"], check=True)
        except:
            pass

        # Step DHYOLO.2: Copy the required folder
        subprocess.run(["cp", "-r", "iTeach/toolkit/iteach_toolkit", "rkit"], check=True)

        # Step DHYOLO.3: Remove the iTeach folder
        subprocess.run(["rm", "-rf", "iTeach"], check=True)

        # Step SAMv2.1: Clone the repository
        samv2_dir = os.path.join(rkit_root_dir, "rkit", "sam2")
        os.makedirs(samv2_dir, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2", samv2_dir], check=True)
        except:
            pass

        # Step SAMv2.2: cd to samv2 and checkout the desired commit branch
        os.chdir(samv2_dir)
        subprocess.run(["git", "checkout", "--branch", "c2ec8e14a185632b0a5d8b161928ceb50197eddc"])

        # Step SAMv2.3: Use sed to comment out line 171 (to get rid of py>=3.10)
        subprocess.run(["sed", "-i", "171s/^/#/", "setup.py"], check=True)

        # Step SAMv2.4: Install samv2
        subprocess.run(["python", "setup.py", "install"], check=True)

        # Step SAMv2.5: move to rkit root directory
        os.chdir(rkit_root_dir)

        # Download GroundingDINO checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            os.path.join(os.getcwd(), "ckpts", "gdino"),
            "gdino.pth"
        )

        # Download MobileSAM checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            os.path.join(os.getcwd(), "ckpts", "mobilesam"),
            "vit_t.pth"
        )

        # Download DHYOLO checkpoints
        dhyolo_checkpoints = [
            "dh-yolo-v1-pb-ddf-524.pt",
            "dh-yolo-exp27-pb-1008.pt",
            "dh-yolo-exp-31-pl-1532.pt",
            "dh-yolo-exp-31-pb-1532.pt"
        ]

        for ckpt in dhyolo_checkpoints:
            self.download_pytorch_checkpoint(
                f"https://huggingface.co/spaces/IRVLUTD/DH-YOLO/resolve/main/pretrained_ckpts/{ckpt}",
                os.path.join(os.getcwd(), "ckpts", "dhyolo"),
                ckpt
            )

        # Download SAM2 checkpoint
        self.download_pytorch_checkpoint(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_large.pth"
        )

        # Download SAM2 checkpoint yaml
        self.download_pytorch_checkpoint(
            "https://raw.githubusercontent.com/facebookresearch/sam2/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_l.yaml"
        )

    def download_pytorch_checkpoint(self, pth_url: str, save_path: str, renamed_file: str):
        """
        Download a PyTorch checkpoint from the given URL and save it to the specified path.
        """
        try:
            import requests
            from tqdm import tqdm  # lazy import

            file_path = os.path.join(save_path, renamed_file)

            # Check if the file already exists
            if os.path.exists(file_path):
                logging.info(f"{file_path} already exists! Skipping download")
                return

            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # Log download attempt
            logging.info("Attempting to download PyTorch checkpoint from: %s", pth_url)

            response = requests.get(pth_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(file_path, 'wb') as file:
                for data in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

            logging.info("Checkpoint downloaded and saved to: %s", file_path)

        except FileNotFoundError as e:
            logging.error("Error: Checkpoint file not found: %s", e)
            raise e


def run_setup():
    setuptools.setup(
        package_dir={"": "rkit"},
        packages=setuptools.find_packages(where="rkit"),
        cmdclass={
            'install': FileFetch,
        },
        package_data={'gdino_cfg': ["rkit/cfg/gdino/GroundingDINO_SwinT_OGC.py"]}
    )


if __name__ == "__main__":
    run_setup()
