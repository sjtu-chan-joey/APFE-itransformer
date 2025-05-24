# APFE-itransformer

**APFE-iTransformer** (Adaptive Pruning Frequency Enhanced iTransformer) is a robust and efficient framework for multivariate time-series forecasting, tailored for noisy and complex environments such as spacecraft telemetry systems. This is the official implementation of our method proposed in:

> 📄 **APFE-iTransformer: Adaptive Frequency-Domain Pruning Enhanced iTransformer for Multivariate Time-Series Forecasting**  
> ✍️ Authors: [Author List]  
> 🏷️ Venue: [Conference or Journal, Year]  
> 📎 [Link to Paper if available]
![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/itrans.png)

---

## 🔍 Overview

APFE-iTransformer addresses two major challenges in time-series forecasting:

- Modeling **long-term temporal dependencies**
- Adapting to **channel-specific noise and spectral redundancy**

It integrates:
- ✅ Legendre Memory Units (LMU) for compact long-term history encoding
  ![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/Legendre.png)
- ✅ Adaptive Top-$k$ Frequency Pruning for denoising
  ![image](https://github.com/sjtu-chan-joey/APFE-itransformer/blob/main/figs/AFPE.png)
- ✅ Inverted Transformer architecture for subsystem-level attention

---

## ✨ Key Features

- 🔄 **Channel-inverted attention**: Models inter-variable dependencies across spacecraft subsystems
- 🧠 **Legendre projection**: Captures long-term memory in a compressed orthogonal basis
- 🎯 **Frequency-domain pruning**: Retains only the most predictive harmonics per channel
- ⚡ **Fast runtime**: Load time ≤ 18s with competitive accuracy  
- 📈 **State-of-the-art performance** on Martian power prediction tasks

---

```
@article{your2024apfe,
  title={APFE-iTransformer: Adaptive Frequency-Domain Pruning Enhanced iTransformer for Multivariate Time-Series Forecasting},
  author={Your Name and Collaborators},
  journal={...},
  year={2024}
}
```

