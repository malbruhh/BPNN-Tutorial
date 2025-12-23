# BankNote Authentication: Neural Network From Scratch

This project implements a **Backpropagation Neural Network (BPNN)** using only `NumPy` to classify banknotes as either **Authentic** or **Forged**. It serves as a practical example of how multi-layer perceptrons learn through gradient descent and matrix calculus without the help of heavy libraries like TensorFlow or PyTorch.

## Project Overview
The model is a 3-layer feedforward network designed to process features extracted from banknote images:
* **Input Layer:** 4 neurons (Variance, Skewness, Curtosis, and Entropy).
* **Hidden Layer:** 4 neurons using the **Sigmoid** activation function.
* **Output Layer:** 1 neuron (Probability: Close to 0 for Fake, Close to 1 for Real).



---

## Key Machine Learning Concepts

### 1. Feedforward & Backpropagation
* **Feedforward:** Data flows from input to output. We use matrix dot products to calculate the "weighted sum" and pass them through a Sigmoid function.
* **Backpropagation:** The "learning" phase. We calculate the error (Difference between prediction and truth) and use the **Chain Rule** to update weights. We start from the output layer and work backward to the input layer.

### 2. Sigmoid Activation & Derivative
We use the Sigmoid function to introduce non-linearity, allowing the network to learn complex patterns:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$ The **Derivative** is crucial for training. It tells the model the "gradient" or slope. If the output is already very close to 0 or 1, the derivative becomes small, telling the model to make only tiny adjustments to the weights.



### 3. Learning Rate ($\alpha$) with Decay
A fixed learning rate can cause a model to "overshoot" the optimal solution. This script implements **Time-Based Decay**:
$$\alpha_{new} = \alpha_{initial} \times \frac{1}{1 + 0.01 \times \text{epoch}}$$ The model takes large steps at the start to find the solution quickly and smaller, "surgical" steps as it gets closer to 100% accuracy to avoid jumping over the perfect weight values.

### 4. Threshold vs. Accuracy
* **Training Threshold (0.01):** During the loop, we are strict. We only count a guess as "Correct" if the raw output is within 0.01 of the target.
* **Final Accuracy:** Calculated at the end by rounding the output to the nearest integer (0 or 1) and comparing it to the test labels.

---

## Evaluation: The Confusion Matrix
The script generates a **Confusion Matrix** to show precisely how the model performed on unseen data:

| Metric | Description |
| :--- | :--- |
| **True Positives (TP)** | Real notes correctly identified as real. |
| **True Negatives (TN)** | Fake notes correctly identified as fake. |
| **False Positives (FP)** | **Type I Error:** Fake notes the model thought were real. |
| **False Negatives (FN)** | **Type II Error:** Real notes the model thought were fake. |



---

## How to Run
1.  Place `BankNote_Authentication.txt` in the same directory as the script.
2.  Install NumPy: 
    ```bash
    pip install numpy
    ```
3.  Run the script:
    ```bash
    python your_script_name.py
    ```

---

## Experiments to Try
* **Hidden Layer size:** Change `weight1` to `(4, 8)` and `weight2` to `(8, 1)`. Does a wider network learn faster?
* **Epoch Count:** Reduce epochs to 500. Is the model still stable?
* **Alpha:** Change `initial_alpha` to `0.5`. Does the accuracy jitter or settle faster?