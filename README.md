# Multiview latent space learning with progressively fine-tuned deep features for unsupervised domain adaptation

## Abstract

Unsupervised Domain Adaptation (UDA) and Multi-source Domain Adaptation (MDA) have emerged as practical techniques to address the domain shift between source and target domains with different statistical distributions, where the target domain often has unlabeled samples. In recent years, end-to-end training approaches have been employed to learn domain-invariant representations, which enable customized adaptations simultaneously for UDA and MDA tasks. Although the conventional pseudo-labeling approach can leverage unlabeled target samples, the potential for inaccurate pseudo-labeling is counterproductive. This work proposes a multiview latent space learning framework with progressively fine-tuned deep features to improve UDA and MDA performance. Specifically, we construct three views, including features directly extracted from pre-trained deep learning models, fine-tuned features with source samples, and fine-tuned features with source samples and pseudo-labeled target samples, to enable unsupervised clustering analysis. More importantly, we utilize a multiview-based selective pseudo-labeling approach that selects the most confident labeled target samples with the maximum conditional probability. Through systematic experiential evaluations incorporating deep learning backbones such as ResNet-50 and DeiT-base, we demonstrate that our proposed multiview latent space learning method consistently outperforms state-of-the-art approaches on various UDA and MDA tasks.


# How to use the code
You need to change the directory to your data.
You can download extracted features used in our experiments from [BaiduPan](https://pan.baidu.com/s/1Glb4KN142kXbz7-BfLggGQ) code: gqtv\
zcy@cczu.edu.cn
# Reference

Zhu, C., Wang, Q., Xie, Y., & Xu, S. (2024). Multiview latent space learning with progressively fine-tuned deep features for unsupervised domain adaptation. Information Sciences, 662, 120223.