# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
# Name:Bhuvaneshwaran H
# Register Number:212223240018
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}

  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop (lig. parameters(),lr=0.001)



# Name:Bhuvaneshwaran H
# Register Number:212223240018
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range (epochs):
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train),y_train)
    loss. backward()
    optimizer.step()
    lig.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

### Dataset Information
Include screenshot of the generated data
<img width="212" height="205" alt="image" src="https://github.com/user-attachments/assets/fadd1779-001f-41f2-b864-777cead0b5e1" />


### OUTPUT

<img width="310" height="178" alt="image" src="https://github.com/user-attachments/assets/53371612-6519-43ba-80ff-d4b197b82126" />

### Training Loss Vs Iteration Plot
Include your plot here
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/3f32691d-6b97-438a-9f8d-57c6390abb6e" />


### New Sample Data Prediction
Include your sample input and output here
<img width="749" height="120" alt="image" src="https://github.com/user-attachments/assets/0095d3c4-1985-40c8-8b93-9b8d63a34d77" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
