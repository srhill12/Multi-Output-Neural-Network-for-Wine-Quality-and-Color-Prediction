

# Multi-Output Neural Network for Wine Quality and Color Prediction

This project implements a multi-output neural network using TensorFlow and Keras to predict wine quality and color based on various chemical properties.

## Dataset

The dataset used for this project is the Wine Quality dataset, which contains chemical properties of different wine samples, along with their quality and color labels. The dataset is sourced from [this link](https://static.bc-edx.com/ai/ail-v-1-0/m19/lesson_3/datasets/wine_quality.csv).

## Data Preprocessing

1. **Quality Encoding**: The `quality` column is one-hot encoded to create three categories: `good`, `ok`, and `bad`.

2. **Color Encoding**: The `color` column is label-encoded where `red` is represented by `0` and `white` by `1`.

3. **Data Splitting**: The dataset is split into features (`X`), quality labels (`y_quality`), and color labels (`y_color`). These are further split into training and testing sets using an 80-20 ratio.

## Model Architecture

A multi-output neural network model is built using TensorFlow's Keras API. The model consists of:

- **Input Layer**: Accepts the input features (11 chemical properties).
- **Shared Hidden Layers**: Two dense layers with ReLU activation that are shared between both outputs.
- **Quality Output Layer**: A dense layer with softmax activation to predict wine quality.
- **Color Output Layer**: A dense layer with sigmoid activation to predict wine color.

### Model Summary
```
Model: "functional"
__________________________________________________________________________________________________
 Layer (type)                     Output Shape          Param #     Connected to                     
==================================================================================================
 input_features (InputLayer)      [(None, 11)]          0                                            
 dense (Dense)                    (None, 64)            768         input_features[0][0]            
 dense_1 (Dense)                  (None, 32)            2080        dense[0][0]                     
 quality_output (Dense)           (None, 3)             99          dense_1[0][0]                   
 color_output (Dense)             (None, 1)             33          dense_1[0][0]                   
==================================================================================================
Total params: 2,980
Trainable params: 2,980
Non-trainable params: 0
__________________________________________________________________________________________________
```

## Model Training

The model is compiled with the following configuration:

- **Optimizer**: Adam
- **Loss Functions**: 
  - Categorical Crossentropy for the quality prediction.
  - Binary Crossentropy for the color prediction.
- **Metrics**: Accuracy for both quality and color predictions.

### Training Output
```
Epoch 1/10
163/163 [==============================] - 5s 7ms/step - color_output_accuracy: 0.7852 - loss: 6.1111 - quality_output_accuracy: 0.6094 - val_color_output_accuracy: 0.9200 - val_loss: 0.9203 - val_quality_output_accuracy: 0.7438
...
Epoch 10/10
163/163 [==============================] - 1s 4ms/step - color_output_accuracy: 0.9556 - loss: 0.7803 - quality_output_accuracy: 0.7532 - val_color_output_accuracy: 0.9454 - val_loss: 0.7683 - val_quality_output_accuracy: 0.7569
```

### Evaluation Results
```
51/51 [==============================] - 0s 2ms/step - color_output_accuracy: 0.9480 - loss: 0.6963 - quality_output_accuracy: 0.7717
```

- **Quality Accuracy**: `77.16%`
- **Color Accuracy**: `95.32%`

These results indicate that the model is performing well on both tasks, particularly in predicting the wine color, which achieved an accuracy of 95.32% on the test data.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    ```
   
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Training Script**:
    ```bash
    python train_model.py
    ```

4. **Evaluate the Model**:
    ```bash
    python evaluate_model.py
    ```

## Next Steps

- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other hyperparameters to improve the model's performance.
- **Data Augmentation**: Implement data augmentation techniques to potentially enhance the model's generalization capabilities.
- **Advanced Architectures**: Explore more complex architectures, such as using convolutional layers for feature extraction or recurrent layers for sequence modeling.
- **Model Deployment**: Deploy the trained model as a web service or API for real-time predictions.

---
