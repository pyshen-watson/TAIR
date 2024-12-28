import imgaug as ia
import imgaug.augmenters as iaa

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Create a degraded dataset by applying image augmentation.")
    parser.add_argument("--src_dir", "-s", type=str, default='data/CUB_200_2011/images', help="The path to the original dataset.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for the random number generator.")
    parser.add_argument("--degrader", "-d", type=str, choices=["fog", "cloud", "rain", "snow"], default="fog", help="The name of the degrader to use.")
    return parser.parse_args()

def get_degrader(name: str) -> iaa.Augmenter:
    
    # Define the degrader
    degraders = {
        "fog": iaa.Fog(),
        "cloud": iaa.Clouds(),
        "rain": iaa.Rain(),
        "snow": iaa.Snowflakes()
    }
    
    return degraders[name]

def create_degraded_dataset(src_dir: Path, dst_dir: Path, degrader: iaa.Augmenter):

    image_paths = list(src_dir.glob("**/*.jpg"))
    
    for image_path in tqdm(image_paths, desc="Creating the degraded dataset"):
        
        # Load the image and degrade it
        image = np.array(Image.open(image_path))
        degraded_image = Image.fromarray(degrader.augment_image(image))
        
        # Save the degraded image
        output_path = dst_dir / image_path.relative_to(src_dir) # Use the same directory structure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        degraded_image.save(output_path)
        
    print(f"Created the degraded dataset at {dst_dir}")
    

if __name__ == '__main__':
    
    args = get_args()
    ia.seed(args.seed)
    
    src_dir = Path(args.src_dir)
    dst_dir = src_dir.parent / f"images_{args.degrader}"
    degrader = get_degrader(args.degrader)
    
    create_degraded_dataset(src_dir, dst_dir, degrader)