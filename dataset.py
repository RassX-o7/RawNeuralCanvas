import os
import numpy as np
from PIL import Image,ImageDraw,ImageTk

class DataSet:
    def __init__(self,mode):
        self.mode=mode
        if self.mode == "train":
            image_path=os.path.join(base,"dataset/train-images.idx3-ubyte")
            label_path=os.path.join(base,"dataset/train-labels.idx1-ubyte")
        elif self.mode == "test":
            image_path=os.path.join(base,"dataset/t10k-images.idx3-ubyte")
            label_path=os.path.join(base,"dataset/t10k-labels.idx1-ubyte")
        with open(image_path,"rb") as images_file:
            header=images_file.read(16) 
            images=np.frombuffer(images_file.read(),dtype=np.uint8)/255
            images=images.reshape(-1,28,28)
        with open(label_path,"rb") as labels_file:
            header=labels_file.read(8)
            labels=np.frombuffer(labels_file.read(),dtype=np.uint8) 
        self.dataset_images=images
        self.dataset_labels=labels
    @staticmethod
    def _augment_og(image:np.ndarray):
        angle=np.random.uniform(-15,15)
        unrot=Image.fromarray((image*255).astype(dtype="uint8"),mode="L")
        new_image=unrot.rotate(angle,resample=Image.BILINEAR,fillcolor=0) 
        return np.array(new_image)/255
    @staticmethod
    def _augment(image: np.ndarray):
        angle = np.random.uniform(-15, 15)
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        rotated = pil_img.rotate(angle,resample=Image.BILINEAR,fillcolor=0)
        img = np.array(rotated) / 255.0
        shift_x = np.random.randint(-2, 3) 
        shift_y = np.random.randint(-2, 3)
        shifted = np.zeros_like(img)
        if shift_y > 0:
            src_y = slice(0, 28 - shift_y)
            dst_y = slice(shift_y, 28)
        else:
            src_y = slice(-shift_y, 28)
            dst_y = slice(0, 28 + shift_y)
        if shift_x > 0:
            src_x = slice(0, 28 - shift_x)
            dst_x = slice(shift_x, 28)
        else:
            src_x = slice(-shift_x, 28)
            dst_x = slice(0, 28 + shift_x)
        shifted[dst_y, dst_x] = img[src_y, src_x]
        return shifted
    def get(self,index,augment=False):
        if augment:
            return DataSet._augment(self.dataset_images[index]),self.dataset_labels[index]
        return self.dataset_images[index],self.dataset_labels[index]
base=os.path.dirname(os.path.abspath(__file__))
train_dataset=DataSet(mode="train")
test_dataset=DataSet(mode="test")