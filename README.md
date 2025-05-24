# AFPE-itransformer

**AFPE-iTransformer** (Adaptive Frequency Pruning Enhanced iTransformer) is a robust and efficient framework for multivariate time-series forecasting, tailored for noisy and complex environments such as spacecraft telemetry systems. This is the official implementation of our method proposed in:

> 📄 **APFE-iTransformer: Adaptive Frequency-Domain Pruning Enhanced iTransformer for Multivariate Time-Series Forecasting**  
> ✍️ Authors: [Joey Chan,Shiyuan Piao,Huan Wang,Zhen Chen,Fugee Tsung,Ershun Pan]  
> 🏷️ Venue: [Paper for SMCA UR]  
> 📎 [Link to Paper after AC]
![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/itrans.png)

---

## 🔍 Overview

AFPE-iTransformer addresses two major challenges in time-series forecasting:

- Modeling **long-term temporal dependencies**
- Adapting to **channel-specific noise and spectral redundancy**

It integrates:
- ✅ Legendre Memory Units (LMU) for compact long-term history encoding
  
  LMU uses orthogonal Legendre polynomials to compress historical input sequences into a fixed-size memory representation.
  This enables the model to preserve long-term temporal dependencies without suffering from gradient decay, making it ideal for capturing trends in power consumption over extended periods.
  ![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/Legendre.png)
- ✅ Adaptive Top-$k$ Frequency Pruning for denoising
  
  This module applies a learnable, per-channel pruning mechanism in the frequency domain.
  By retaining only the top-$k$ spectral components, it removes noise while preserving the most predictive patterns.
  This greatly enhances robustness under real-world signal perturbations (e.g., solar flares, subsystem instability).
  ![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/AFPE.png)
- ✅ Inverted Transformer architecture for subsystem-level attention
  
We extend the iTransformer structure by (1) transposing the input to treat each channel (i.e., subsystem) as an attention token, enabling cross-subsystem dependency modeling;
and (2) applying a learnable Softmax excitation within each channel to highlight key temporal features. This dual-level attention improves both accuracy and interpretability in spacecraft telemetry forecasting.

---

```
@article{your2024apfe,
  title={AFPE-iTransformer: Adaptive Frequency-Domain Pruning Enhanced iTransformer for Multivariate Time-Series Forecasting},
  author={Your Name and Collaborators},
  journal={...},
  year={2024}
}
```
## Contributors
If you have any questions or want to use the code, feel free to contact:

Chan.Joey (SJTU_Chan_Joey@outlook.com)

Zhen Chen (chenzhendr@sjtu.edu.cn)

