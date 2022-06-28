# Improve-NLP-models-by-adding-Adversarial-Training
Adversarial Training, Min-Max, Fast Gradient Method, Gradient Penalty

## Data source:
IFLYTEK (CLUE Benchmark) Dataset: 

https://metatext.io/redirect/iflytek-(clue-benchmark)

TNEWS (CLUE Benchmark) Dataset: 

https://metatext.io/redirect/toutiao-text-classification-for-news-titles-(tnews)-(clue-benchmark)

## Abstract
The goal is to use adversarial training to improve the model for better understanding the true meaning of the context.

## Introduction
For the confrontation in deep learning, generally it has two meanings. One is Generative Adversarial Network, GAN, the other is Adversarial Training, which focuses on the model’s robustness under small perturbation.

In recent years, along with the development of deep learning, adversarial examples have gotten more and more attention. In the CV field, we need to enhance the robustness of the model through adversarial attacks and defenses against the model. For example, we have to prevent the model from recognizing red light into green light because of some random noize in the autopilot system. In the NLP field, similar adversarial training also exists, but the adversarial training in NLP is more like a regulation method to improve the generalization ability of the model.
