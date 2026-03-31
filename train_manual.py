import numpy as np
import math
import random

# 1. 神经网络
class TrainableNet:
    def __init__(self, input_size=34, learning_rate=0.01):
        self.lr = learning_rate
        
        # He Initialization
        self.W1 = np.random.randn(input_size, 256) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, 256))
        
        self.W2 = np.random.randn(256, 128) * np.sqrt(2.0/256)
        self.b2 = np.zeros((1, 128))
        
        self.W3 = np.random.randn(128, 1) * np.sqrt(2.0/128)
        self.b3 = np.zeros((1, 1))

        self.a3 = None

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    def sigmoid_derivative(self, x): s = self.sigmoid(x); return s * (1 - s)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backward(self, X, y_true):
        self.forward(X)
        m = X.shape[0]
        
        error = self.a3 - y_true
        delta3 = error * self.sigmoid_derivative(self.z3)
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(self.z2)
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        
        dW3 = np.dot(self.a2.T, delta3) / m
        db3 = np.sum(delta3, axis=0, keepdims=True) / m
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        return np.mean(np.abs(error))

    def save_weights(self, filename="model_weights.npz"):
        np.savez(filename, W1=self.W1, b1=self.b1.flatten(), W2=self.W2, b2=self.b2.flatten(), W3=self.W3, b3=self.b3.flatten())
        print(f"权重已保存至 {filename}")


# 2. 数据生成器
def generate_eagle_eye_data(samples=10000):
    print(f"正在生成 {samples} 条数据...")
    X_data = []
    Y_data = []
    
    CENTER_X = 4.75 / 2.0  
    CENTER_Y = 38.0        
    HOUSE_R = 1.83
    
    for _ in range(samples):
        num_stones = random.randint(1, 10)
        state_vec = np.zeros(34)
        
        my_min_dist = 100.0
        opp_min_dist = 100.0
        
        focus_house = random.random() < 0.95
        
        for i in range(num_stones):
            if focus_house:
                sx = random.uniform(CENTER_X - 2.0, CENTER_X + 2.0)
                sy = random.uniform(CENTER_Y - 3.0, CENTER_Y + 3.0)
            else:
                sx = random.uniform(0, 4.75)
                sy = random.uniform(0, 44.5)
            
            dx = sx - CENTER_X
            dy = sy - CENTER_Y
            state_vec[i*2] = dx
            state_vec[i*2+1] = dy
            
            dist = math.sqrt(dx**2 + dy**2)
            is_my_stone = (i % 2 == 0)
            
            if is_my_stone:
                if dist < my_min_dist: my_min_dist = dist
            else:
                if dist < opp_min_dist: opp_min_dist = dist

        state_vec[32] = 1.0 
        state_vec[33] = 1.0 
        
        
        label = 0.2
        
        if my_min_dist < opp_min_dist:
            base_win = 0.6
            
            bonus = 0.4 * np.exp(-(my_min_dist**2) / (0.8**2)) 
            
            label = base_win + bonus
        else:
            label = 0.1
            
        if my_min_dist > HOUSE_R and opp_min_dist > HOUSE_R:
            label = 0.5
            
        X_data.append(state_vec)
        Y_data.append([label])
        
    return np.array(X_data), np.array(Y_data)

# 3. 训练循环
def create_batches(X, y, batch_size=32):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx : start_idx + batch_size]
        yield X[excerpt], y[excerpt]

if __name__ == "__main__":
    X_train, y_train = generate_eagle_eye_data(20000) 
    
    net = TrainableNet(learning_rate=0.05) 
    
    epochs = 200 
    batch_size = 64
    
    print(f"开始训练 (Batch Size={batch_size})...")
    
    for i in range(epochs):
        batch_losses = []
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            loss = net.backward(X_batch, y_batch)
            batch_losses.append(loss)
        
        avg_loss = np.mean(batch_losses)
        print(f"Epoch {i}: Loss = {avg_loss:.5f}")
            
    print("训练完成！")
    net.save_weights()