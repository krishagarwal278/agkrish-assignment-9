# **Neural Network Visualization and Analysis**

## **Overview**
This project implements and visualizes a simple feedforward neural network with one hidden layer for binary classification. The neural network is trained on a synthetic 2D dataset, and various aspects of its learning process are visualized, including:
- Hidden layer features and their transformations.
- Decision boundaries in the input space.
- Gradients with visual representations of their magnitudes.

## Submission
Video demo: 

## **Key Features**
1. **Neural Network Architecture**:
   - **Input Layer**: 2 neurons (2D input data).
   - **Hidden Layer**: 3 neurons with configurable activation functions (`tanh`, `relu`, `sigmoid`).
   - **Output Layer**: 1 neuron for binary classification.

2. **Visualization**:
   - **Hidden Space Features**: 3D scatter plot of the hidden layer activations.
   - **Decision Boundary**: Contour plot of the decision boundary in the input space.
   - **Gradients**: Edge thickness visualizes gradient magnitudes between neurons.

3. **Activation Functions**:
   - Supports `tanh`, `relu`, and `sigmoid` activation functions, allowing comparison of their effects on learning and visualization.

4. **Animation**:
   - Dynamic visualization of the training process, saved as GIFs.

---

## **How to Run**

## Part 0: Setup Environment

You can use the `Makefile` to install all dependencies. In your terminal, simply run:

```bash
make install
```

This will automatically install the necessary packages listed in `requirements.txt`, including:

- flask
- numpy
- scikit-learn
- scipy
- matplotlib

## Part 1: Implementing Feedforward Neural Network and Visualization

1. **Build the Feedforward Neural Network**: 
   - Implement the Feedforward Neural Network from scratch with a forward function for forward propagation and a backward function for backpropagation. Implement three activation functions: 'tanh', 'relu', 'sigmoid'.
  
2. **Visualization**
   - Implement the visualization code to plot 
     - The learned features, distorted input space and decision hyperplane in the hidden space
     - The decision boundary in the input space
     - The gradients where the edge thickness visually represents the magnitude of the gradient.
     - Create an animation illustrating the entire training process.

   Here's a basic example of the visualization. It's a simplified version, so feel free to enhance it.
   ![til](example-output/visualize.gif)

  

## Part 2: Testing Your Code with a Static Input (Optional)

1. If you prefer, you can also test the code locally by running the script directly and specifying necessary parameters. 

2. Run the script in your terminal:
   
   python neural_networks.py

3. Check the output in the `results` folder.

## Part 3: Running the Interactive Module

Once the environment is set up, you can start the Flask application by running:

```bash
make run
```

This will start the Flask server and make the interactive application available locally at `http://127.0.0.1:3000`.

1. Open your browser and go to `http://127.0.0.1:3000`.
2. Choose the corresponding parameters and click "Train and Visualize". 
3. The resulting figure will be displayed. It may take a while before the results show up.


