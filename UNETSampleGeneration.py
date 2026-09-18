from Image import DiffusionImage
from Generator import Generator

N_SAMPLES = 6

generator = Generator(name="tiny-dppm-resnet")
images = DiffusionImage.generate(generator, 1262, N_SAMPLES)
images.save_all(".",prefix="generation-")
