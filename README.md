# LLMB4ABSC

This repository contains the full implementation of the Large Language Model Bootstrapping for Aspect-Based Sentiment Classification (LLMB4ABSC) framework, developed as part of the MSc thesis in Econometrics & Management Science (Business Analytics & Quantitative Marketing track) at the Erasmus School of Economics.

All experiments are run in Google Colab using an NVIDIA T4 GPU and LLMs are loaded via [Hugging Face](https://huggingface.co/) using the [Unsloth](https://github.com/unslothai/unsloth) library for efficient inference.
---

## Introduction
LLMB4ABSC is an unsupervised framework that leverages LLMs for both synthetic data generation and sentiment classification in ABSC. The first stage, IDG4ABSC, generates synthetic aspect-sentiment-sentence triplets. The second stage, LLM4ABSC, uses the self-generated data as few-shot In-Context Learning (ICL) examples to classify sentiment.

## Overview

### *Main Algorithm*
- The `IDG4ABSC.py` script 
- The `LLM4ABSC.py` script 
---

