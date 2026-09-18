import os
import re

from Image import DiffusionImage
from Generator import Generator

N_SAMPLES = 6

VERSIONS_DIR = "versions"
PREFIX = "tiny-dppm-resnet-"

versions = []

for name in os.listdir(VERSIONS_DIR):
    match = re.fullmatch(rf"{re.escape(PREFIX)}(\d+)", name)
    if match:
        versions.append((int(match.group(1)), name))

if not versions:
    raise FileNotFoundError(
        f"No versions matching '{PREFIX}<version>' found in {VERSIONS_DIR}/"
    )

version, model_name = max(versions)

print(f"Using {model_name}")

generator = Generator(name=model_name)

images = DiffusionImage.generate(generator, 6, N_SAMPLES)
images.save_all(".", prefix="generation-")