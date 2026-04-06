# ASL-CNN
ASL Sign Language Classification — CNN
Convolutional neural network for classifying American Sign Language hand signs, achieving 93.22% test accuracy across 24 letter classes.
MSDS 458 — Artificial Intelligence and Deep Learning, Northwestern University (Winter 2026).

Dataset
Sign Language MNIST — Kaggle
34,627 grayscale 28x28 images across 24 ASL letter classes (J and Z excluded — require motion). Data files not included; download from Kaggle.

Results
ModelAccuracyF1 (Macro)EpochsBaseline (2 blocks, dropout=0.50)0.92610.92397Depth variant (3 blocks, dropout=0.50)0.92530.91695Dropout=0.25 (2 blocks)0.91490.90817Dropout=0.75 (2 blocks) — BEST0.93220.92977
Best model precision: 0.9395 | Recall: 0.9292
Top 5 confused letter pairs: N→M (66), I→Y (42), N→A (35), W→V (30), G→T (22) — all reflect genuine visual similarity in ASL hand shapes.

Key Findings

Higher dropout improved performance — overfitting was the main challenge at this dataset size
Added depth (3 blocks) did not help — insufficient spatial resolution at 28x28 to benefit
Batch norm anomaly in Epoch 1 (33% val accuracy) recovered fully by Epoch 2


Architecture
Input (28x28x1)
→ Conv2D(32, 3x3, ReLU) + BatchNorm + MaxPool
→ Conv2D(64, 3x3, ReLU) + BatchNorm + MaxPool
→ Flatten → Dense(128, ReLU) → Dropout(0.75)
→ Dense(24, Softmax)
Stack: Python, TensorFlow/Keras, scikit-learn, Google Colab (T4 GPU)
