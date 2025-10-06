from torch import nn 
import torch

class svm:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'

class knn:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'
    
class decision_tree:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'
    
class random_forest:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'
    
class xgboost:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'
    
class lightgbm:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'
    
class catboost:
    def __init__(self):
        pass

    def train(self, x, y):
        '''
        x: training data
        y: training labels
        '''

    def save(self, filepath):
        print("Saving model to:", filepath)

    def load(self, filepath):
        print("Loading model from:", filepath)

    def evaluate(self, x, y):
        ''''
        x: test data
        y: test labels
        '''
        return "{accuracy:, precision:, recall:, f1-score:}"
    
    def predict(self, input_data):
        print("Predicting with model for input:", input_data)
        return '0 or 1'

class Conv(nn.Module):
    def __init__(
            self, 
            in_channels, out_channels, 
            dim,
            kernel_size=3, padding=1, stride=2
        ):
        super(Conv, self).__init__()
        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.batch_norm = nn.BatchNorm1d(out_channels)
        elif dim == 2:
             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
             self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))
     
class Exodia(nn.Module):
    def __init__(self):
        super(Exodia, self).__init__()
        # Define layers here
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

    def train_model(self, train_loader, criterion, optimizer, epochs=5):
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        print("Model saved to:", filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
        print("Model loaded from:", filepath)

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return {"accuracy": accuracy}

    def predict(self, input_data):
        with torch.no_grad():
            outputs = self.forward(input_data)
            _, predicted = torch.max(outputs.data, 1)
        return predicted