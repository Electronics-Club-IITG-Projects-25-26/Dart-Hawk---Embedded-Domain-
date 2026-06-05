# DartHawk — Embedded Domain

## About the Project

**DartHawk** is an autonomous Nerf turret built as a hands-on embedded systems project that combines real-time computer vision with a servo-actuated pan–tilt mechanism. The main idea is simple: a camera observes the scene, the software identifies and follows a target, and the turret continuously adjusts its **yaw (left–right)** and **pitch (up–down)** so it stays aimed at the target as it moves.

This project is focused on making the full tracking-and-actuation loop work smoothly in real time. Instead of being a purely “vision-only” demo, DartHawk is meant to feel like an integrated system: the vision pipeline produces a stable target estimate, and the embedded control side translates that information into consistent mechanical motion. The result is a platform that can be used to experiment with tracking approaches, control logic, and integration between software and hardware.

## What DartHawk Does

DartHawk is designed to:

- Capture a live video feed from a camera
- Detect and track a target in the frame using **OpenCV**
- Continuously update turret orientation based on the target’s position
- Drive servo motors to align the turret in real time

The “embedded domain” part of the project covers the electronics and control side of the turret—servo actuation, motion constraints, and the logic that turns tracking output into actual turret movement—while the computer vision side handles target tracking and coordination.

## Technology Used

This repository mainly uses:

- **Python** for the computer vision pipeline (OpenCV) and higher-level coordination
- **C++** for embedded/firmware-side control and actuator handling

## Background

This repository is part of the **Electronics Club IITG** project work for the **2025–26** cycle. The goal is to build a clean, working turret platform that demonstrates practical integration of computer vision with embedded control in a real hardware system.
