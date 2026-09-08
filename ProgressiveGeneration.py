from Image import DiffusionImage
from Generator import Generator

N_SAMPLES = 1

generator = Generator()

def generate_gif(epoch=545, N=1, filename="diffusion.gif", duration=50):
    generator.load(epoch)

    frames = []

    for tensors in generator.generate_iter(N):
        # tensors: [N, C, H, W]
        for tensor in tensors:
            frame = DiffusionImage(tensor).getAsPIL()
            frames.append(frame)

    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )


def generate_stages(epoch=545, N=1, prefix="diffusion", nthsave=50):
    generator.load(epoch)

    step = 0

    for tensors in generator.generate_iter(N):
        if step % nthsave == 0:
            for i, tensor in enumerate(tensors):
                frame = DiffusionImage(tensor).getAsPIL()
                frame.save(f"{prefix}_{i}_step{step}.png")

        step += 1



