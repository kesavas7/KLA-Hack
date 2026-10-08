# RCAN — Residual Channel Attention Network (KLA AI Hackathon Image Restoration)

An end-to-end deep learning pipeline built for **Single-Channel (Grayscale) Image Super-Resolution and Denoising** targeting semiconductor microscopy patterns for the **KLA AI Hackathon** (March 14–18, 2026)[cite: 2].

---

## Challenge Overview & Objective

The objective of this challenge is to restore single-channel images degraded by **speckled noise, blur, and low spatial resolution**, reconstructing them back to near-original quality at $256 \times 256$ spatial resolution[cite: 2]. 

Solutions are benchmarked on both restoration quality and inference optimization:
- **Primary Metrics:** Peak Signal-to-Noise Ratio (**PSNR**) and Structural Similarity Index Metric (**SSIM**).
- **Key Challenges:** Preserving fine-grain semiconductor edge features, handling varying degradation severity, and running optimized inference within Kaggle resource limits[cite: 2].

---

## Model Architecture & Specs

- **Model:** Residual Channel Attention Network (RCAN) right-sized for small/medium scientific datasets[cite: 2].
- **Parameters:** ~3.86M–4.09M trainable parameters[cite: 2].
- **Configuration:** 5 Residual Groups $\times$ 10 Residual Channel Attention Blocks (RCABs) = 50 total RCABs[cite: 2].
- **Feature Dimension (`n_feats`):** 64 channels[cite: 2].
- **Reduction Ratio (`reduction`):** 16 (Squeeze-and-Excitation style Channel Attention)[cite: 2].
- **Upscaling Scale Factor:** $2\times$ (PixelShuffle from $128 \times 128$ patches to $256 \times 256$ outputs)[cite: 2].

---

## Key Pipeline Features & Innovations

- **Hybrid Reconstruction Loss:** $0.7 \times \text{Charbonnier Loss} + 0.3 \times (1 - \text{SSIM})$ balancing pixel-level fidelity and structural edge preservation[cite: 2].
- **Data Pipeline:** Direct `float32` $[0, 1]$ range tensor processing with random sub-patch extraction ($64\times64$ LR to $128\times128$ HR) and spatial augmentations (horizontal/vertical flips, $90^\circ$ rotations)[cite: 2].
- **Optimization Strategy:** Adam optimizer with linear warmup (5 epochs) and Cosine Annealing learning rate decay down to $1\text{e-}6$[cite: 2].
- **Validation & Early Stopping:** PSNR evaluation on a 10% validation split after every epoch, saving best checkpoints with a patience threshold of 25 epochs[cite: 2].
- **Inference Strategy:** $8\times$ Test-Time Augmentation (x8 TTA) combining spatial flips and rotational symmetry transformations with inverse mapping[cite: 2].
- **Submission Serialization:** Converts output arrays into `.npy` files and serializes them into a base64-encoded `submission.csv` compliant with competition submission rules[cite: 2].

---

## Directory Structure & Submission Format

### Submission Output Specifications
1. Reads noisy LR input images from `/kaggle/input/competition-data/test/NoisyLR/`.
2. Output shape per image: 2D NumPy array ($256 \times 256$), `float32` in $[0, 1]$ range[cite: 2].
3. Individual predictions saved to `/kaggle/working/submission/*.npy`[cite: 2].
4. Serialized base64 submission generated at `/kaggle/working/submission.csv`[cite: 2].

```text
/kaggle/working/
├── best_rcan.pth         # Best checkpoint state dict
├── best_metrics.json     # Log of best epoch and validation PSNR
├── submission/           # Individual .npy predictions (256x256 float32)
│   ├── 000000.npy
│   └── ...
└── submission.csv        # Final submission CSV containing id & base64 encoded npy
