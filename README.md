# 🛡️ Generative Adversarial Networks (GANs) in Cybersecurity

This repository supports **Part B** of my final year BSc Cybersecurity project. The project explores the use of GAN-based models to generate synthetic intrusion detection samples using the CIC-IDS-2017 dataset. The goal is to augment imbalanced classes, evaluate realism, and assess ethical concerns in dual-use scenarios.

> 🔗 GitHub Repository: https://github.com/AlekseyTsar3vi4/COMP60003-GANS

---

## 📁 Repository Structure

| File / Folder              | Description                                                             |
|---------------------------|-------------------------------------------------------------------------|
| `notebooks.zip`           | Google Colab notebooks for training CGAN, WGAN-GP, and IGAN-style models |
| `images.zip`              | Visualisation outputs: PCA, t-SNE, ROC, and model architecture diagrams |
| `models.zip`              | Trained model files (PyTorch `.pt`) for inspection or reuse             |
| `final_cgan_ready_scaled.zip` | Preprocessed CIC-IDS-2017 dataset (compressed due to GitHub size limits) |
| `IGAN_Attempt3_Synthetic.csv` | Output synthetic samples from IGAN-style model for attack-only generation |

---

## 📒 Google Colab Notebooks

Click below to open each training notebook directly in Colab:

- [Attempt 1 – CGAN (Colab)](https://colab.research.google.com/drive/1_TaRHeA53EFNxZn7U7j-FGfQQwboyOM0?usp=sharing)  
- [Attempt 2 – WGAN-GP (Colab)](https://colab.research.google.com/drive/1d3HmdsoerXQrDVn2dBYs6FToJQiFRs8r?usp=sharing)  
- [Attempt 3 – IGAN-Style (Colab)](https://colab.research.google.com/drive/1ERO5GbAEkonfLCEsSjlMHKHB4ZqfoFSE?usp=sharing)

> 📌 These notebooks include the training pipeline, logging, visualisation, and model saving.

---

## 🧪 Evaluation Summary of GAN Attempts

| Attempt  | Model Type   | Focus         | AUC   | Synth vs Real Acc. | Silhouette Score | F1 Δ     | Key Observations                                      |
|----------|--------------|---------------|-------|---------------------|------------------|----------|--------------------------------------------------------|
| Attempt 1| CGAN         | All Classes   | 0.949 | 83.43%              | 0.4838           | -0.0001  | Best cluster alignment, stable training, robust output |
| Attempt 2| WGAN-GP      | All Classes   | 0.563 | 63.63%              | 0.0028           |  0.0000  | Low fidelity, poor cluster structure, weak signal      |
| Attempt 3| IGAN-style   | Attacks Only  | 0.747 | 79.00%              | 0.3967           | +0.0001  | Strong class mimicry, suited for adversarial defence   |

## 🏗️ Architecture Summary of Each GAN Attempt

| Aspect                   | Attempt 1 (CGAN)             | Attempt 2 (WGAN-GP)             | Attempt 3 (IGAN-style)          |
|--------------------------|------------------------------|----------------------------------|---------------------------------|
| Focus                    | All classes (Normal + Attacks) | All classes (Normal + Attacks) | Attack classes only (1–4)       |
| GAN Type                 | Vanilla CGAN                 | WGAN-GP                          | Conditional GAN (Targeted)      |
| Noise Dimension          | 100                          | 100                              | 100                             |
| Generator Depth          | 2 layers (256 → output)      | 3 layers (256 → 512 → output)    | 3 layers (128 → 256 → output)   |
| Discriminator Depth      | 3 layers (256 → 128 → 1)     | 3 layers (512 → 256 → 1)         | 3 layers (256 → 128 → 1)        |
| Activation (G)           | ReLU → Sigmoid               | ReLU (no final activation)       | ReLU (no final activation)      |
| Activation (D)           | LeakyReLU                    | LeakyReLU                        | LeakyReLU → Sigmoid             |
| Loss Function            | BCE + Label Smoothing        | Wasserstein + Gradient Penalty   | BCE                             |
| Conditioning             | One-hot label concatenation  | One-hot label concatenation      | One-hot label concatenation     |
| Training Ratio           | 1:1 G/D updates              | 1:5 G:D updates                  | 1:1 G/D updates                 |
| Final Output Scaling     | Sigmoid                      | None                             | None                            |
| Class Balance Handling   | Balanced                     | Balanced                         | Attack-focused only             |


Each attempt included:
- PCA and t-SNE visualisation
- ROC curves (Discriminator AUC)
- Silhouette and class clustering metrics
- Classifier performance (F1, binary accuracy)

---

## 🧠 Key Techniques

- Conditional GANs (CGAN) for label-based synthesis  
- WGAN-GP for training stability and gradient penalty  
- IGAN-style minority class focus for attack augmentation  
- Evaluation via dimensionality reduction (PCA, t-SNE) and classifiers

---

## 📎 Appendices (linked in academic report)

- **Appendix A** – Preprocessing visual screenshots (e.g. label encoding, scaling)
- **Appendix B** – GitHub project link and direct Colab access

---

## 📌 Requirements

To run notebooks locally or in Colab:

```txt
torch==2.1.0
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.1
scikit-learn==1.2.2
tqdm==4.65.0
seaborn==0.12.2
