# Sentiment Analysis Service with PyTorch and FastAPI

A production-style sentiment analysis service built using a PyTorch LSTM model with explicit negation handling. The system exposes a FastAPI endpoint, supports containerized deployment, and demonstrates end-to-end ML inference from model loading to HTTP response.

## Motivation

This project was built to understand the full lifecycle of an ML model in production, including training, inference, API design, and deployment. The primary focus is on building a deployable and maintainable ML service rather than optimizing for state-of-the-art accuracy.

## Limitations and Future Improvements

- Replace LSTM with transformer-based models for improved contextual understanding
- Add batch inference and asynchronous processing for higher throughput
- Introduce monitoring for model drift and inference latency
- Compare performance against pretrained transformer baselines

## What This Project Does Not Do

This project does not aim to compete with modern transformer-based sentiment models. Instead, it serves as a foundation for understanding model deployment, inference pipelines, and ML system design.

## System Architecture

![Architecture Diagram](diagram.png)
