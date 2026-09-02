from UNET import UNET
from Image import DiffusionImage
from Generator import Generator

N_SAMPLES = 6

generator = Generator()
images = DiffusionImage.generate(generator, 545, 6)
images.save_all(".",prefix="generation-")
