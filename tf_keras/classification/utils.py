from PIL import Image
import os

def png2jpg(data_dir, rm = False):

    for label_name in os.listdir(data_dir):
        if os.path.isdir('/'.join([data_dir, label_name])):
            for img_name in os.listdir('/'.join([data_dir, label_name])):
                if img_name.endswith('png'):
                    im1 = Image.open('/'.join([data_dir, label_name, img_name]))
                    if im1.mode in ("RGBA", "P"):
                        im1 = im1.convert("RGB")
                    im1.save('/'.join([data_dir, label_name, img_name[:-4] + '.jpg']))

                    if rm:
                        os.remove('/'.join([data_dir, label_name, img_name]))

if __name__ == "__main__":
    png2jpg(r'C:\Users\mehra\OneDrive\Desktop\deep_learning\data\BrainTumor', rm=True)



