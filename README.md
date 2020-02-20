# maniSourceLocalize

This repository contains information and some implemented procedures to estimating the position of an audio source based on a semi-supervised manifold learning technique. In particular, we try to estimate the position of a source while a microphone array may be moving or corrupt (e.g. low battery). This requires being able to estimate when an array is moving, what array is moving, and how to update our learned position estimator that was trained on data that reflects the original microphone array positions.

To find the working demo, navigate to code\sourceLocalization\sourceLocal_main.m. To run the file a mex compiler is needed. Information regarding the procedure is commented in the file, and more information can be found in LaTex-Node Movement Detection in the LaTex file.
