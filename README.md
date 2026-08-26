<div align="center">
  <img src="assets/output.gif"/>
</div>
<br>
    
# TinyDPPM

A small, from-scratch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in PyTorch.

TinyDPPM isn't trying to be fast or state-of-the-art — it's built for learning and experimentation. Every core piece of a diffusion model is implemented by hand: the noise scheduler, the forward (noising) process, a small U-Net, the training loop, and the reverse (sampling) process.

## How it works

<img src="assets/output.png"/>

1. **`NoiseScheduler`** precomputes the beta/alpha/alpha-bar schedule used throughout training and sampling, in either `linear` or `cosine` mode.
2. **`ForwardDiffusion`** uses that schedule to noise a clean image to timestep `t` (`getNoisyTensor`), and to reverse a single denoising step given the model's predicted noise (`reverse`).
3. **`UNET`** is a small convolutional U-Net (3 encoder blocks, a bottleneck, 3 decoder blocks with skip connections) that predicts the noise added at a given timestep. The timestep is turned into a sinusoidal embedding, passed through an MLP, and injected into every block via a learned linear projection.
4. **`TrainProcess`** runs mini-batch gradient descent: sample a random `t` per image, noise the batch, predict the noise with the U-Net, and minimize MSE against the real noise. Loss is printed per batch, color-coded (green/red/yellow) depending on whether it improved.
5. **`VersionManager`** checkpoints the model to `versions/<name>-<epoch>.pth` after every epoch, and can reload the latest checkpoint on startup so training resumes where it left off. It also has a `save_on_fail` decorator that checkpoints the model if training crashes or is interrupted (e.g. `Ctrl+C`).
6. **`Datasets`** wraps `torchvision` datasets behind a small, consistent interface (`train_loader` / `test_loader`). It currently supports **MNIST, CIFAR-10, and Flowers102**, downloading datasets automatically when needed.
7. **`Image` (`DiffusionImage`)** is a thin wrapper around a tensor/PIL image that handles the back-and-forth conversion used throughout the pipeline.

Training and sampling are built out of these same components.

## Project layout

| File | Purpose |
|---|---|
| `DPPM.py` | Entry point for training the diffusion model — running this file starts model training |
| `Generation.py` | Entry point for sampling — runs the full reverse diffusion process and saves `output.png` |
| `NoiseScheduler.py` | Beta/alpha/alpha-bar schedule (linear or cosine) |
| `ForwardDiffusion.py` | Forward noising (`q`) and single-step reverse (`p`) diffusion math |
| `VarationalAutoEncoder.py` | Simple VAE used to encode images into a lower-dimensional latent space — running this file starts VAE training |
| `UNET.py` | The noise-prediction model, with timestep embeddings |
| `TrainProcess.py` | Training loop (mini-batch gradient descent, loss logging) |
| `VersionManager.py` | Checkpoint saving/loading, crash-safe autosave |
| `Datasets.py` | `MNIST` / `CIFAR-10` / `Flowers102` dataset + dataloader wrappers |
| `Image.py` | `DiffusionImage` tensor/PIL image helper |
| `util.py` | Small terminal color-printing helper for loss values |

## Requirements

- Python 3
- `torch`
- `torchvision`
- `matplotlib`
- `pillow`
- `numpy`

```bash
pip install torch torchvision matplotlib pillow numpy
```

## TensorBoard

TinyDPPM uses **TensorBoard** to visualize training losses.

The training code writes TensorBoard event files to:

```text
runs/<model-name>/
```

TensorBoard is kept in a **separate virtual environment** to avoid dependency conflicts with the main training environment.

Two helper scripts are provided:

| Script | Purpose |
|---|---|
| `createtensorboard-env.sh` | Creates the isolated TensorBoard virtual environment and installs TensorBoard |
| `tensorboard.sh` | Activates the TensorBoard environment and starts TensorBoard |

### Setup

Run this once:

```bash
./createtensorboard-env.sh
```

### Start TensorBoard

In a separate terminal from your training process:

```bash
./tensorboard.sh
```

TensorBoard will monitor the `runs/` directory and display the training losses as they are logged.

The training process and TensorBoard can run simultaneously in separate terminals.

For example:

**Terminal 1:**

```bash
python DPPM.py
```

**Terminal 2:**

```bash
./tensorboard.sh
```

Then open the address printed by TensorBoard in your browser.