# LLMB4ABSC

This repository contains the full implementation of the Large Language Model Bootstrapping for Aspect-Based Sentiment Classification (LLMB4ABSC) framework, developed as part of the MSc thesis in Econometrics & Management Science (Business Analytics & Quantitative Marketing track) at the Erasmus School of Economics.

All experiments are run in Google Colab using an NVIDIA T4 GPU and LLMs are loaded via [Hugging Face](https://huggingface.co/) using the [Unsloth](https://github.com/unslothai/unsloth) library for efficient inference.

---

## Introduction
This project introduces LLMB4ABSC, a two-stage unsupervised framework to ABSC that leverages LLMs for both data generation and sentiment classification in ABSC. The first stage, IDG4ABSC, generates and filters aspect-sentiment-sentence triplets from the aspect terms in the test set. The second stage, LLM4ABSC, uses the self-generated data as few-shot In-Context Learning (ICL) examples to improve their own ABSC performance. The SemEval 2014, 2015, and 2015 datasets are used for model evaluation. Results show that LLMB4ABSC significantly improves its zero-shot ABSC performance in terms of accuracy and F1-score on the evaluated LLMs: Mistral, LLaMa, and Gemma. Results show that LLMB4ABSC consistently improves zero-shot performance across all models in terms of accuracy and F1 score. In addition, IDG4ABSC proves to be a strong Data Augmentation (DA) method, yielding performance gains when fine-tuning traditional supervised ABSC models.

## Features
- Combines LLM-based DA and LLM-based ABSC into one of the first fully unsupervised ABSC frameworks.
- Evaluated on SemEval 2014, 2015, and 2016 datasets, showing consistent improvements over zero-shot baselines across three LLMs.
- Introduces IDG4ABSC, a new DA framework that outperforms other LLM-based methods on state-of-the-art ABSC models.
- Proposes a novel quality filtering pipeline that enhances performance in both few-shot ABSC and fine-tuned supervised models.

### *Main Algorithm*
- The `IDG4ABSC.py` script 
- The `LLM4ABSC.py` script 
---

