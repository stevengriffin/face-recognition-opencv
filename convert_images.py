from PIL import Image
import glob

root_dir = 'unprocessed_dataset/'
for filename in glob.iglob(root_dir + '**/*.pgm', recursive=True):
    Image.open(filename).save(filename[:-3] + "png")
