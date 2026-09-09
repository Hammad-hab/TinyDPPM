# TinyDPPM

A from-scratch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in PyTorch, trained on the [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset at 256x256 resolution. The repo also includes standalone VAE and RGB-delta autoencoder experiments used for latent-space and image-space compression research.

![Forward diffusion example](assets/output.png)

## What's in here

The core pipeline is a conditional U-Net that predicts the noise added to an image at a given timestep, trained with the standard DDPM objective (`MSE(predicted_noise, true_noise)`), plus a full checkpointing/versioning system and generation utilities (single image, batch, and progressive GIF).

| File | Role |
|---|---|
| `NoiseScheduler.py` | Builds the beta/alpha/alpha-bar schedules (`linear` or `cosine`) used to control how much noise is added at each timestep. |
| `ForwardDiffusion.py` | Forward process q(x_t \| x_0) (adds noise per the schedule) and the reverse/denoising step p(x_{t-1} \| x_t) used at sampling time. |
| `UNET.py` | The noise-prediction network. 6 encoder blocks, 2 bottleneck conv layers, 6 decoder blocks with skip connections, and sinusoidal timestep embeddings injected at every block via learned linear projections. |
| `TrainProcess.py` | Training loop: mini-batch gradient descent over the dataset, AdamW optimizer, cosine-annealed LR, TensorBoard logging, checkpointing each epoch. |
| `VersionManager.py` | Handles saving/loading model+optimizer checkpoints by epoch, auto-resuming from the latest one, and marking/recovering from failed epochs (`.f.pth` files) if training crashes mid-epoch. |
| `Datasets.py` | Thin wrappers around `torchvision.datasets` for MNIST, CIFAR-10, and Flowers102 with DataLoader creation. |
| `Generator.py` | Loads a trained checkpoint and runs the full reverse diffusion loop (T to 0) to produce new images from pure noise. |
| `Image.py` | `DiffusionImage`/`DiffusionImages` helper classes wrapping tensors for easy conversion to PIL, saving, and viewing. |
| `ProgressiveGeneration.py` | Renders the entire denoising trajectory as a GIF or as saved intermediate-step PNGs. |
| `UNETSampleGeneration.py` | Quick script to generate and save a batch of samples from a given checkpoint. |
| `DPPM.py` | Main entry point -- wires everything together and runs training. |
| `DRGBAutoEncoder.py` | Separate experiment: a small conv autoencoder that predicts an RGB *delta* to reconstruct warped/masked flower images (not part of the diffusion pipeline). |
| `VarationalAutoEncoder.py` | Separate experiment: a beta-VAE (conv encoder/decoder) for Flowers102, with a live-tunable beta read from the `beta_read` file each epoch. |
| `transforms.py` | Custom image transforms: Perlin noise generator, `PetalSelection` (radial masking), `ImageWarp` (smooth random displacement via grid_sample) -- used to pretrain/stress-test the autoencoders. |
| `util.py` | Device auto-detection (CUDA -> MPS -> CPU) and colored loss-delta printing for terminal logs. |
| `tensorboard.sh` / `createtensorboard-env.sh` | Helper scripts to set up and launch a TensorBoard viewer for the `runs/` logs. |

## How the diffusion model works here

**Forward process.** Given a clean image `x0` and a schedule computed once up front, `ForwardDiffusion.getNoisyTensor` samples a random noise `eps` and directly produces the noisy image at any timestep `t` in closed form:

```
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
```

**Training.** For each batch, a random timestep `t` is sampled per image, the image is noised to `x_t`, and the U-Net is asked to predict the `eps` that was added -- not the original image. Loss is plain MSE between predicted and true noise.

**Reverse process (sampling).** Starting from pure Gaussian noise, `Generator.generate` steps `t` from `T` down to `1`. At each step the U-Net predicts the noise in the current `x_t`, `ForwardDiffusion.reverse` uses that to estimate `x0`, computes the posterior mean/variance of q(x_{t-1} | x_t, x0), and samples the next, slightly-less-noisy image (deterministic mean-only at `t=1`).

**Timestep conditioning.** The U-Net doesn't take `t` as an extra image channel -- it computes a sinusoidal embedding of `t` (dimension 128), passes it through a small MLP, then projects it to each block's channel count and adds it in as a bias at every encoder/bottleneck/decoder stage. This is the same trick used in the original DDPM/Stable Diffusion U-Nets, just without attention layers or residual blocks (this is a lightweight/"tiny" version).

**Schedules.** Two options are implemented in `NoiseScheduler`:
- `linear` -- betas spaced linearly between `1e-4` and `0.02` (original DDPM paper default).
- `cosine` -- the improved schedule from Nichol & Dhariwal, which adds noise more gradually near `t=0` and `t=T`.

`DPPM.py` uses `cosine` with `T=1000`.

## Checkpointing

`VersionManager` saves both model and optimizer state to `versions/<name>-<epoch>.pth` after every epoch. If training is interrupted (`KeyboardInterrupt` or any exception) inside a function wrapped with `@vm.save_on_fail`, it saves a `.f.pth` "failed epoch" checkpoint instead, so the next run knows to redo that epoch rather than silently resuming from stale weights. `load_latest` scans the versions directory, finds the highest epoch, and resumes training from there automatically -- this is what lets `DPPM.py` be re-run indefinitely (e.g. across Colab sessions) without manual bookkeeping.

## Running it

Dependencies (not pinned in the repo -- install as needed): `torch`, `torchvision`, `matplotlib`, `numpy`, `pillow`, `tensorboard`.

```bash
# Train the diffusion model (resumes automatically if versions/ has checkpoints)
python3 DPPM.py

# Generate a batch of samples from a specific epoch checkpoint
python3 UNETSampleGeneration.py

# Render the full denoising trajectory as a GIF
python3 ProgressiveGeneration.py

# Inspect the dataset / forward-noising process directly
python3 Datasets.py flowers
python3 ForwardDiffusion.py   # saves output.png showing an image at increasing noise levels
python3 NoiseScheduler.py     # sanity-check assertions on the schedule math

# View training curves
bash createtensorboard-env.sh   # one-time env setup
bash tensorboard.sh
```

`Generator`, `UNETSampleGeneration.py`, and `ProgressiveGeneration.py` currently hard-code the checkpoint name `tiny-dppm` and default to `epoch=545` -- change these to match whatever epoch you've actually trained to.

## Notes / things to know before extending this

- **Hard-coded resolution.** The U-Net and dataset transforms assume 256x256x3 input; changing resolution means adjusting the `cblocks` channel schedule and the encoder/decoder chain in `UNET.py`.
- **`ForwardDiffusion.reverse` clamps `x0_pred` to `[-3, 3]`**, with a comment noting this range is meant for VAE latent stats rather than raw pixels -- this is a leftover from an abandoned latent-diffusion experiment (see below) and may need adjusting to `[-1, 1]`/`[0, 1]` for pixel-space use, depending on how you normalize images.
- **VAE / DRGBAutoEncoder are not wired into the diffusion pipeline.** `TrainProcess._getVAE` exists and is called out in a comment (`# with torch.no_grad(): mu, _ = self.vae.encoder(x0)`) but is commented out -- the current pipeline operates directly in pixel space, not latent space. These two files represent a parallel, unfinished attempt at latent diffusion.
- **No attention blocks.** The U-Net is convolution-only; this keeps it lightweight but limits its ability to model long-range spatial dependencies compared to the attention-augmented U-Nets used in most modern diffusion papers.
- **`DISABLE_LOGS` in `TrainProcess.py`** is `True` by default, so per-batch console logs are suppressed; epoch-level TensorBoard logging still runs regardless.