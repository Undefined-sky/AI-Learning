"""Record an image and scalar values in TensorBoard.

Run without data, or provide a local image with --image.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional local image; otherwise generate a demo image.")
    parser.add_argument("--logdir", type=Path, default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()

    if args.image is not None:
        if not args.image.is_file():
            parser.error(f"Image not found: {args.image}")
        with Image.open(args.image) as image:
            image_array = np.asarray(image.convert("RGB"))
    else:
        image_array = np.zeros((64, 64, 3), dtype=np.uint8)
        image_array[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
        image_array[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None] * 4

    with SummaryWriter(str(args.logdir)) as writer:
        writer.add_image("demo/image", image_array, 1, dataformats="HWC")
        for step in range(100):
            writer.add_scalar("demo/y_equals_x", step, step)
    print(f"TensorBoard logs written to: {args.logdir}")


if __name__ == "__main__":
    main()
