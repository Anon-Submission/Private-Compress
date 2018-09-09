# Private-Compress

Demo codes for the AAAI'19 submission *Private Model Compression via Knowledge Distillation*

## Prerequisites

1. Performance test

    - Linux or macOS
    - NVIDIA GPU + CUDA CuDNN 8.0 or CPU(not recommend)
    - Tensorflow-gpu 1.3.0, keras 2.0.5, python 3.6, numpy 1.14.0, scikit-learn 0.18.1

2. Implementation on Android

    - Linux or macOS
    - JDK 1.8
    - Android Studio 2.3.3
    - Android SDK 7.0, Android SDK Build Tools 26.0.1, Android SDK Tools 26.1.1, Android SDK Platform Tools 26.0.1

## Notes

`student_model.py` and `teacher_model.py` are the network classes of student model and teacher model, respectively.

`teacher_convlarge_cifar.npy` stores the weights of the teacher model pretrained on both the public data and the sensitive data of CIFAR-10.

`teacher_convlarge_public.npy` stores the weights of the teacher model pretrained on the public data of CIFAR-10. It is used to generate adaptive norm bound.

`private-compress-cifar.py` is an example of RONA which trains a compact neural network on CIFAR-10.

TFDroid is a demo project on Android system for testing the time overhead of Large-Conv neural network on mobile devices.
