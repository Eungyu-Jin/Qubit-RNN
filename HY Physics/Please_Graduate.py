import numpy as np
import math
from qutip import create, basis, sigmaz, sigmax, sigmay
import matplotlib.pyplot as plt


class quantum_system():
    def __init__(self, omega_r = 2*np.pi*0.80, kai = -2*np.pi*0.18, qubit_initial = 0, cavity_initial = 1, superposition= False):
        self.planck = 1.0545718*10**(-34)
        self.omega_r = omega_r
        self.kai = kai
        self.kappa = 2*np.pi*(7.2) #MHz * time = t
        # dt = 1e-3 : 1e-9 actual
        # times = n * dt
        self.n_cavity = 10
        self.ad = create(self.n_cavity)
        self.a = self.ad.dag()

        self.qubit_initial = qubit_initial
        self.cavity_initial = cavity_initial
        self.superposition = superposition

        # Hamiltonian
        self.h_int = np.kron(self.kai*0.5*(self.ad*self.a), sigmaz())
        self.h_r = np.kron(np.identity(self.n_cavity),(0.5)*self.omega_r*sigmax())
        self.h = self.h_int + self.h_r
        
        # initial state
        if self.qubit_initial ==0:
            self.qubit_state = basis(2, 1)
        else:
            self.qubit_state = basis(2, 0)

        if self.superposition == True:
            self.cavity_state = (basis(self.n_cavity, self.cavity_initial) + basis(self.n_cavity,self.cavity_initial+1))/np.sqrt(2)
        else:
            self.cavity_state = basis(self.n_cavity, self.cavity_initial)

        self.psi0 = np.kron(self.cavity_state, self.qubit_state)
    
    def exp_H(self, t):
        from scipy import linalg
        return linalg.expm(-1j*self.h*t)
    
    def bra_m(self, c, q):
        if q == 0:
            q_state = 1
        else:
            q_state = 0
            
        return np.kron(basis(self.n_cavity, c), basis(2, q_state))
        
    def prob(self, c, q,t):
        return (abs(np.dot(self.bra_m(c, q).T, np.dot(self.exp_H(t), self.psi0))))**2

    def monte_carlo(self, n, n_sample, t_i, t_f):
        import random
        import itertools
        #prob_index = [(i,j) for i,j in itertools.product(range(self.n_cavity),range(2))]
        outcome = np.zeros((self.n_cavity * 2, n))
        dt = (t_f - t_i)/(n-1)

        t = t_i
        for z in range(n):
            prob_list = [self.prob(i,j,t).reshape(1)[0] for i,j in itertools.product(range(self.n_cavity),range(2))]
            prob_line = np.cumsum(prob_list)
            for _ in range(n_sample):
                random_num = random.random()
                for k in range(len(prob_line)):
                    if k == 0 and random_num < prob_line[k]:
                        outcome[k, z] += 1
                    elif prob_line[k-1] <= random_num and random_num < prob_line[k]:
                        outcome[k, z] += 1
                t += dt
        
        relative_prob = outcome/n_sample

        import pandas as pd
        df = pd.DataFrame(relative_prob.T)
        df.index = np.linspace(t_i,t_f,n_sample)

        return df   
    
    def preprocess(self, df, split_ratio, time_step):
        if self.superposition==True:
            df = df.iloc[:,self.cavity_initial*2: (self.cavity_initial + 2)*2]
        else:
            df = df.iloc[:,self.cavity_initial*2: (self.cavity_initial + 1)*2]
        
        split = int(len(df)*split_ratio)
        train = df.iloc[:split]
        test = df.iloc[split:]

        def dataset(data, window_size):
            X_list, y_list = [], []
            for i in range(len(data) - window_size):
                X_list.append(np.array(data.iloc[i:i+window_size,:]))
                y_list.append(np.array(data.iloc[i+window_size, :]))
            return np.array(X_list), np.array(y_list) 

        train_X, train_y = dataset(train, time_step)
        test_X, test_y = dataset(test, time_step)

        return train_X, train_y, test_X, test_y

class Model():
    def __init__(self, df, train_feature, train_label, test_feature, test_label):
        self.df = df
        self.train_feature, self.train_label, self.test_feature, self.test_label = train_feature, train_label, test_feature, test_label

    def LSTM(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional

        model = Sequential()
        model.add(LSTM(256,activation='tanh',input_shape=(self.train_feature.shape[1],self.train_feature.shape[2]), dropout=0.2))
        #model.add(Dropout(0.2))
        model.add(Dense(self.train_feature.shape[2], activation='linear'))
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=[tf.keras.metrics.BinaryCrossentropy()])

        return model
    
    def fit(self, model, epochs = 100, batch_size = 128, show_loss = True):
        from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
        
        history = model.fit(self.train_feature, self.train_label, batch_size = batch_size ,epochs=epochs,validation_split=0.2, verbose=1, callbacks = [early_stopping])
        
        if show_loss:
            plt.figure(figsize=(12,8))
            plt.plot(history.history['loss'], label = 'loss')
            plt.plot(history.history['val_loss'], label = 'val_loss')
            plt.legend()
            plt.title('Model Loss')
            plt.show()
        return model, history
    
    def predict(self, model, show_plot = True):
        predict = model.predict(self.test_feature)

        from sklearn.metrics import mean_squared_error, r2_score
        print('rmse: {}'.format(np.sqrt(mean_squared_error(self.test_label, predict))))
        print('r2: {}'.format(r2_score(self.test_label, predict)))

        if show_plot:
            plt.figure(figsize=(12,8))
            plt.subplot(211)
            plt.plot(self.df.index[-self.test_label.shape[0]:],predict)
            plt.title("Predict")

            plt.subplot(212)
            plt.plot(self.df.index[-self.test_label.shape[0]:], self.test_label)
            plt.title("Test Label")
            plt.show()

