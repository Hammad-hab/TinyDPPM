import torch
from PIL import Image
from torchvision import transforms

from DRGBAutoEncoder import DRGBAutoEncoder
from VersionManager import VersionManager


model = DRGBAutoEncoder()

vm = VersionManager(
    model,
    "tiny-drgbae",
    dir="versions/tiny-drgbae/"
)

vm.load_latest(True, True)

model.eval()

image = Image.open("generation-0.png").convert("RGB")

to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)

image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():
    delta = model(image_tensor)

    output = image_tensor + delta
    output = output.clamp(0, 1)

output_image = transforms.ToPILImage()(output.squeeze(0))
output_image.save("output.png")
