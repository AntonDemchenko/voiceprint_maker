Voiceprint maker with SincNet architecture and CosFace head implemented using TensorFlow v2+.

# Usage
1. Download dataset with voice records and speaker labels.
2. Split dataset into parts by creating three files containing record paths. These files would be used for training, validating and testing respectively. Record path should be relative to the dataset root.
3. Create .npy file which contains dictionary mapping record path to speaker label. Speaker label should be integer from range [0, C - 1] where C is number of distinct speakers.
4. Create configuration file like cfg/SincNet_TIMIT.cfg.
5. Train: ```python train.py --cfg=<your configuration file path>```.
6. Test: ```python test_print_maker --cfg=<your configuration file path>```.
7. Make voiceprints: ```python wav_to_voiceprint.py --cfg=<your configuration file path>```.

If you work with TIMIT dataset then you can skip 1-4 steps and use cfg/SincNet_TIMIT.cfg as configuration file.

## References
[1] SincNet original code written in PyTorch by the autor (https://github.com/mravanelli/SincNet)

[2] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)

[3] Hao Wang, Yitong Wang and others, "CosFace: Large Margin Cosine Loss for Deep Face Recognition" [Arxiv](https://arxiv.org/pdf/1801.09414.pdf)

[4] CosFace repository (https://github.com/4uiiurz1/keras-arcface)