# SenseStick Navigation Aid

**Prototype code and project materials associated with the published SenseStick assistive-system research.**

SenseStick is an edge-AI assistive platform for visually impaired and elderly users, combining navigation support, road-safety perception, local-language accessibility, biomedical sensing and caregiver-oriented monitoring.

## Publication

This project led to the first-author paper:

**A. R. Aditta, P. K. Das, U. R. Arshee and A. B. M. Aowlad Hossain, “SenseStick: A Bangla-English Edge-AI Cane for Road-Safety and Biomedical Reliability Monitoring,” IEEE BECITHCON 2026.**

The published system integrates Bangla-English contextual feedback, camera-based perception, OCR, QR/barcode and color utilities, staircase/road-sign/zebra-crossing perception, biomedical monitoring, fall/stick-recovery alerting, IoT caregiver support and a TinyML-compatible biomedical reliability gate.

## Repository scope and publication evolution

This repository contains **project/prototype code from the SenseStick development line**. It should not be interpreted as a byte-for-byte software release of every module reported in the final BECITHCON 2026 paper.

Some repository files reflect earlier prototype hardware/software choices, while the published system reports the later integrated research configuration, including **MAX30102 physiological sensing, MPU6050 motion sensing and ESP32 caregiver communication**. The paper also evaluates a TinyML biomedical reliability gate for distinguishing stable from motion-affected physiological windows before caregiver notification.

This distinction is intentional: the repository documents the engineering project lineage, while the peer-reviewed paper is the authoritative source for the final reported architecture, experimental protocol and quantitative results.

## Core project capabilities

- Camera-based object detection and navigation feedback
- Direction and approximate distance assistance
- Bangla and English audio feedback
- OCR-based text reading
- QR/barcode and color utilities
- Road-safety perception modules
- Physiological sensing and caregiver-oriented monitoring
- Motion/fall-related safety logic

## Prototype software / methods represented here

`Python` · `C++` · `YOLOv8` · `OpenCV` · `OCR` · `TensorFlow` · `Arduino`

## Repository guide

- [`Object Detect with Direction, Distance and Audio Feedback.py`](Object%20Detect%20with%20Direction,%20Distance%20and%20Audio%20Feedback.py) — object-detection and navigation feedback pipeline
- [`Bangla lan.py`](Bangla%20lan.py) — Bangla-language support
- [`OCR.py`](OCR.py) — OCR-based text-reading component
- [`Colour Detection.py`](Colour%20Detection.py) — color-detection utility
- [`QR CODE.py`](QR%20CODE.py) — QR-code processing utility
- [`model/`](model/) — model assets used by the project
- [`Report_ECE_3200.pdf`](Report_ECE_3200.pdf) — earlier project report

## Academic context

SenseStick represents my broader engineering background in embedded systems, computer vision, sensing and applied machine learning. My current primary research directions are RF/mmWave IC design and VLSI/EDA.

[GitHub Profile](https://github.com/AryanAditta) · [Google Scholar](https://scholar.google.com/citations?user=Id4ByyUAAAAJ&hl=en) · [RF/mmWave Research](https://aryanaditta.github.io/rfic-academic/) · [VLSI/EDA Research](https://aryanaditta.github.io/eda-academic/)
