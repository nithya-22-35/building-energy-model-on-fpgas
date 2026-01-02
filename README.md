# building-energy-model-on-fpgas
Quantized neural network for building energy usage prediction deployed on FPGA using QKeras and hls4ml (PYNQ-Z2).
# Building Energy Model on FPGAs

This project demonstrates deployment of a quantized neural network for
building energy usage prediction on FPGA using QKeras and hls4ml.

The model is trained in TensorFlow, quantized using QKeras, converted
to High-Level Synthesis (HLS) using hls4ml, and deployed on a PYNQ-Z2 FPGA.

---

## Dataset
- Source: Kaggle – Building Energy Usage Dataset
- Type: Time-series building energy data
- Target: Energy usage (regression)

---

## Model
- Fully-connected neural network
- Quantized weights and activations (6-bit)
- Regression output (linear activation)

---

## FPGA Flow
1. Dataset preprocessing and training (QKeras)
2. Model pruning and quantization
3. hls4ml conversion
4. Vivado HLS and bitstream generation
5. On-board inference using PYNQ-Z2

---

## Tools & Frameworks
- TensorFlow / Keras
- QKeras
- hls4ml
- Vivado
- PYNQ-Z2

---

## Status
Work in progress 🚧
