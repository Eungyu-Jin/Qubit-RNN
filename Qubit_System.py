import numpy as np
from qutip import create, basis, sigmaz, sigmax, sigmay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, concatenate, LSTM, GRU, Dense, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam


class Quantum_System():
    def __init__(self, qubit_initial = 0, cavity_initial = 1, superposition= False):
        self.planck = 1.0545718*10**(-34)
        self.omega_r = 2*np.pi*0.80
        self.kai = -2*np.pi*0.18
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
            #self.cavity_state = (basis(self.n_cavity, self.cavity_initial) - basis(self.n_cavity,self.cavity_initial+1))/np.sqrt(2)
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

    def monte_carlo(self, n_sample, n_times, t_i, t_f):
        import random
        import itertools
        #prob_index = [(i,j) for i,j in itertools.product(range(self.n_cavity),range(2))]
        experiment = np.zeros((self.n_cavity * 2, n_times))
        theory = []
        dt = (t_f - t_i)/(n_times-1)

        t = t_i
        for z in range(n_times):
            prob_list = [self.prob(i,j,t).reshape(1)[0] for i,j in itertools.product(range(self.n_cavity),range(2))]
            theory.append(prob_list)
            prob_line = np.cumsum(prob_list)
            for _ in range(n_sample):
                random_num = random.random()
                for k in range(len(prob_line)):
                    if k == 0 and random_num < prob_line[k]:
                        experiment[k, z] += 1
                    elif prob_line[k-1] <= random_num and random_num < prob_line[k]:
                        experiment[k, z] += 1
                t += dt
        
        relative_prob = experiment/n_sample

        import pandas as pd
        df_experiment = pd.DataFrame(relative_prob.T)
        df_theory = pd.DataFrame(theory)
        df_experiment.index = np.linspace(t_i,t_f,n_sample)
        df_theory.index = np.linspace(t_i,t_f,n_sample)

        return df_experiment, df_theory   
    
    def slice_df(self, df_experiment, df_theory):
        if self.superposition:
            df_experiment_slice = df_experiment.iloc[:,self.cavity_initial*2: (self.cavity_initial + 2)*2]
            df_theory_slice = df_theory.iloc[:,self.cavity_initial*2: (self.cavity_initial + 2)*2]
        else:
            df_experiment_slice = df_experiment.iloc[:,self.cavity_initial*2: (self.cavity_initial + 1)*2]
            df_theory_slice = df_theory.iloc[:,self.cavity_initial*2: (self.cavity_initial + 1)*2]
        
        return df_experiment_slice, df_theory_slice

    def preprocess(self, df, time_step):
        split_ratio = 0.8
        split = int(len(df)*split_ratio)
        train = df.iloc[:split]
        test = df.iloc[split:]

        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        scaler = MinMaxScaler()
        train_scale = scaler.fit_transform(train) 
        test_scale = scaler.transform(test)

        def create_dataset(dataset, window_size):
            X, y = [], []
            for i in range(len(dataset) - window_size):
                X.append(np.array(dataset[i:i+window_size,:]))
                y.append(np.array(dataset[i+window_size, :]))
            return np.array(X), np.array(y) 

        train_X, train_y = create_dataset(train_scale, time_step)
        test_X, test_y = create_dataset(test_scale, time_step)

        return train_X, train_y, test_X, test_y, scaler

class Models():
    def __init__(self,qunatum_system, df_experiment, df_theory):
        self.df_experiment, self.df_theory = qunatum_system.slice_df(df_experiment, df_theory)
        self.train_X, self.train_y, self.test_X, self.test_y, self.scaler = qunatum_system.preprocess(df=self.df_experiment, time_step=10)

        self.dropout_rate = 0.2
        self.units = 64
        self.input_shape = (self.train_X.shape[1],self.train_X.shape[2])

    def LSTM(self):
        model = Sequential()
        model.add(LSTM(self.units,activation='tanh',input_shape=(self.train_X.shape[1],self.train_X.shape[2]), dropout=self.dropout_rate))
        #model.add(Dropout(0.2))
        model.add(Dense(self.train_X.shape[2], activation='sigmoid'))
        model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=[BinaryCrossentropy()])
        return model

    def BiLSTM(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.units,activation='tanh',input_shape=self.input_shape, dropout=self.dropout_rate)))
        #model.add(Dropout(0.2))
        model.add(Dense(self.train_X.shape[2], activation='sigmoid'))
        model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=[BinaryCrossentropy()])
        return model

    def transformer_encoder(self, inputs, key_dim, num_heads, ff_dim):
        # Normalization and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=self.dropout_rate)(x, x)
        x = Dropout(self.dropout_rate)(x)
        res = x + inputs

        # Feed Forward
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = SpatialDropout1D(self.dropout_rate)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def Transfomer_conv(self, key_dim, num_heads, ff_dim, num_blocks):
        inputs = Input(shape=self.input_shape)
        x = Conv1D(filters=self.units, activation = 'relu', kernel_size=self.train_X.shape[1], padding='causal', dilation_rate = 1)(inputs)
        x = SpatialDropout1D(self.dropout_rate)(x)
        #x = Conv1D(filters=self.units, activation = 'relu', kernel_size=self.train_X.shape[1], padding='causal', dilation_rate = 2)(x)
        #x = SpatialDropout1D(self.dropout_rate)(x)
        #x = Conv1D(filters=self.units, activation = 'relu', kernel_size=self.train_X.shape[1], padding='causal', dilation_rate = 4)(x)
        #x = SpatialDropout1D(self.dropout_rate)(x)

        for _ in range(num_blocks):
            x = self.transformer_encoder(x, key_dim, num_heads, ff_dim)

        avg_pool  = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        
        outputs = Dense(self.train_X.shape[2], activation='sigmoid')(conc) 
        model = Model(inputs, outputs)
        #model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanSquaredError()])
        model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=[BinaryCrossentropy()])

        return model

    def fit(self, model, epochs = 100, batch_size = 128, verbose = 1, show_loss = True):
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        early_stopping = EarlyStopping(monitor='val_loss', verbose=verbose, patience=50)
    
        history = model.fit(self.train_X, self.train_y, batch_size = batch_size ,epochs=epochs,validation_split=0.2, verbose=verbose, callbacks = [early_stopping])
        train_loss = model.evaluate(self.train_X, self.train_y, verbose = 0)[0]
        print('train loss: {}'.format(train_loss))

        if show_loss:
            plt.figure(figsize=(12,8))
            plt.plot(history.history['loss'], label = 'loss')
            plt.plot(history.history['val_loss'], label = 'val_loss')
            plt.legend()
            plt.title('Model Loss')
            plt.show()
        return model
    
    def predict(self, model, show_plot = True):
        predict = model.predict(self.test_X)
        test_loss = model.evaluate(self.test_X, self.test_y, verbose = 0)[0]
        print('test loss: {}'.format(test_loss))
        
        from sklearn.metrics import mean_squared_error, r2_score
        print('rmse: {}'.format(np.sqrt(mean_squared_error(self.test_y, predict))))
        print('r2: {}'.format(r2_score(self.test_y, predict)))

        test_y = self.scaler.inverse_transform(self.test_y)
        predict = self.scaler.inverse_transform(predict)

        #cavity = int(self.df_experiment.columns[0]/2)
        #qubit = 0
        if len(self.df_experiment.columns) == 2:
            #labels = ["cavity {} & qubit {}".format(cavity, qubit), "cavity {} & qubit {}".format(cavity, qubit+1)]
            labels = ["|0>", "|1>"]
        else:
            #labels = ["cavity {} & qubit {}".format(cavity, qubit), "cavity {} & qubit {}".format(cavity, qubit+1), "cavity {} & qubit {}".format(cavity+1, qubit), "cavity {} & qubit {}".format(cavity+1, qubit+1)]
            labels = ["|00>", "|01>", "|10>", "|11>"] 

        if show_plot:
            plt.figure(figsize=(14,8))
            plt.subplot(311)
            plt.plot(self.df_experiment.index[-test_y.shape[0]:],predict)
            plt.title("Predict")
            plt.xlabel("time")
            plt.ylabel("probabilty")
            plt.legend(loc = 'upper right', labels = labels)

            plt.subplot(312)
            plt.plot(self.df_experiment.index[-self.test_y.shape[0]:], test_y)
            plt.title("Test Experiment")
            plt.xlabel("time")
            plt.ylabel("probabilty")
            plt.legend(loc = 'upper right', labels = labels)

            plt.subplot(313)
            plt.plot(self.df_theory.index[-self.test_y.shape[0]:], self.df_theory.iloc[-self.test_y.shape[0]:].values)
            plt.title("Test Theory")
            plt.xlabel("time")
            plt.ylabel("probabilty")
            plt.legend(loc = 'upper right', labels = labels)

            plt.tight_layout()
            plt.show()
        
        return predict

class Experiment():
    def __init__(self, qubit_initial = 0, cavity_initial=1,superposition=False, n_samples=100, n_times=100, t_i=0, t_f=1):
        self.omega_r = 2*np.pi*0.80
        self.kai = -2*np.pi*0.18

        #model fit setting
        self.time_step = 10
        self.qubit_initial = qubit_initial
        self.cavity_initial = cavity_initial
        self.superposition = superposition
        self.n_samples = n_samples
        self.n_times = n_times
        self.t_i, self.t_f = t_i, t_f

        #transformer setting
        self.key_dim = 16
        self.num_heads = 2
        self.ff_dim = 64
        self.num_blocks = 1

    def Run_system(self, model = 'BiLSTM', epochs=100, batch_size =256):
        qs = Quantum_System(qubit_initial=self.qubit_initial, cavity_initial=self.cavity_initial, superposition=self.superposition)
        df_experiment, df_theory = qs.monte_carlo(n_sample=self.n_samples,  n_times=self.n_times, t_i=self.t_i, t_f=self.t_f)
        Model = Models(qunatum_system=qs, df_experiment = df_experiment, df_theory=df_theory)
        if model == 'LSTM':
            rnn = Model.LSTM()
        elif model == 'BiLSTM':
            rnn = Model.BiLSTM()
        elif model == 'Transformer_conv':
            rnn = Model.Transfomer_conv(key_dim=self.key_dim, num_heads=self.num_heads, ff_dim = self.ff_dim, num_blocks=self.num_blocks)
        else:
            print('Error')

        rnn = Model.fit(rnn,epochs=epochs, batch_size = batch_size, verbose=0, show_loss = False)
        _ =  Model.predict(rnn,show_plot=True)
