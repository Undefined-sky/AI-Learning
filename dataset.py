from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import argparse
class MyDataset(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=sorted(
            name for name in os.listdir(self.path)
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label=self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read local ants/bees images with a custom Dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "dataset_" / "train")
    args = parser.parse_args()
    for label_dir in ("ants_image", "bees_image"):
        if not (args.root / label_dir).is_dir():
            parser.error(f"Missing directory: {args.root / label_dir}. Supply your local data with --root.")
        dataset = MyDataset(str(args.root), label_dir)
        print(f"{label_dir}: {len(dataset)} images")
        if len(dataset):
            image, label = dataset[0]
            print(f"First sample: size={image.size}, mode={image.mode}, label={label}")
            image.close()
