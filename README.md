# SenseStick Navigation Aid

**Edge-AI assistive system for visually impaired and elderly users, combining navigation support, safety monitoring and Bangla-English accessibility features.**

SenseStick integrates embedded sensing, computer vision and audio feedback in a single assistive platform. The project includes real-time object detection, distance/direction feedback, OCR, Bangla-English speech support, health sensing and fall-detection functionality.

## Core Capabilities

- Real-time object detection with audio feedback.
- Direction and distance assistance for nearby objects.
- Bangla and English text-to-speech support.
- OCR-based document/text reading.
- Pulse-rate and SpO₂ monitoring.
- Fall detection and alerting.
- Color and QR-code detection utilities.

## Hardware Platform

- Raspberry Pi
- Arduino Nano
- MPU6050 motion sensor
- MAX30100 pulse/SpO₂ sensor
- ESP8266 Wi-Fi module
- Camera module
- Buzzer, earphones and rechargeable power source

## Software / Methods

`Python` · `C++` · `YOLOv8` · `OpenCV` · `OCR` · `TensorFlow` · `Arduino` · `ESP8266`

## Repository Guide

- [`Object Detect with Direction, Distance and Audio Feedback.py`](Object%20Detect%20with%20Direction,%20Distance%20and%20Audio%20Feedback.py) — object-detection and navigation feedback pipeline.
- [`Bangla lan.py`](Bangla%20lan.py) — Bangla-language support.
- [`OCR.py`](OCR.py) — OCR-based text-reading component.
- [`Colour Detection.py`](Colour%20Detection.py) — color-detection utility.
- [`QR CODE.py`](QR%20CODE.py) — QR-code processing utility.
- [`model/`](model/) — model assets used by the project.
- [`Report_ECE_3200.pdf`](Report_ECE_3200.pdf) — project report.

## Publication

This project led to the first-author paper:

**“SenseStick: A Bangla-English Edge-AI Cane for Road-Safety and Biomedical Reliability Monitoring,” IEEE BECITHCON 2026.**

## Academic Context

The project reflects my broader engineering background in embedded systems, computer vision, sensing, applied machine learning and assistive technology.
