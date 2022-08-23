import streamlit as st
from Modell.ResNet_1DCNN import ResNet
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
import glob
import pickle
import matplotlib.pyplot as plt
import wfdb as wf
import os
import BaselineWanderRemoval
from scipy.signal import butter, lfilter, decimate


import plotly.express as px

## def
def Create_the_model(a= 0):

    ## config:
    length = 1250  # Length of each Segment
    model_name = 'ResNet18_LSTM'  # DenseNet Models
    model_width = 64  # Width of the Initial Layer, subsequent layers start from here
    num_channel = 1  # Number of Input Channels in the Model
    problem_type = 'Regression'  # Classification or Regression
    output_nums = 2  # Number of Class for Classification Problems, always '1' for Regression Problems

    filepath = 'Model/Final_ResNet_LSTM_1_hubber_loss.hdf5'

    ## Model
    Model = ResNet(length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='max',
                   dropout_rate=0.3).ResNet_18_LSTM()
    # load weights
    Model.load_weights(filepath)
    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-5)
    loss = tf.keras.losses.Huber(delta=1)
    # Compile model (required to make predictions)

    Model.compile(opt, loss=loss , metrics=['mae'])
    return Model

###
class DataGenerator_old_data(Sequence):
    def __init__(self,
                 data_paths,
                 subject, state='Train',
                 batch_size=8,
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        self.batch_size = batch_size
        self.data_path = data_paths
        self.subject = subject
        self.state = state
        self.n_channels = n_channels
        self.dim = None
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.X_indexes = None
        self.X_data = None
        self.y_data = None
        self._Create_large_data()
        self.on_epoch_end()

    def _Create_large_data(self):
        ##### create the large data set
        data_file = glob.glob(self.data_path + "*.csv")
        df_original = pd.DataFrame()
        for i in data_file:
            df = pd.read_csv(i)
            df = df.dropna()
            df_original = pd.concat([df_original, df], axis=0)

        #df_original = df_original[df_original['DBP'] > 40]
        df_original = df_original.dropna()
        df_final = df_original[df_original['Name'] == self.subject]
        self.X_data = df_final.iloc[:, 5:].values # 5
        self.y_data = df_final.iloc[:, 3:5].values # 3 :5
        self.dim = self.X_data.shape[1]
        print(df_final.shape)

    def __len__(self):
        # if self.X is not None:
        #   'Denotes the number of batches per epoch'
        #   self.X_indexes = np.arange(self.X.shape[0])
        #   return int(np.floor(self.X.shape[0] / self.batch_size))
        # else:
        #   self._Creat_large_data(self)
        #   self.X_indexes = np.arange(self.X.shape[0])
        return int(np.floor(self.X_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.X_indexes = np.arange(self.X_data.shape[0])
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temps = [self.X_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_data.shape[0])
        if self.shuffle ==False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        # self._Create_large_data(self)
        X = np.empty((self.batch_size, self.dim))
        y = []
        for i, ID in enumerate(list_IDs_temps):
            X[i,] = self.X_data[ID]
            X = X.astype('float32')
            y.append(self.y_data[ID])
        X = X[:, :, np.newaxis]
        y = np.array(y, dtype=np.float32)
        return X, y


class DataGenerator_new_data(Sequence):
    def __init__(self,
                 data_paths,file_name,state='Train',
                 batch_size=8,
                 n_channels=1,
                 n_classes=2,
                 shuffle=True):
        self.batch_size = batch_size
        self.data_path = data_paths
        self.file_name = file_name
        self.state = state
        self.n_channels = n_channels
        self.dim = None
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.X_indexes = None
        self.X_data = None
        self.y_data = None
        self._Create_large_data()
        self.on_epoch_end()

    def _Create_large_data(self):
        ##### create the large data set
       # data_file = glob.glob(self.data_path +self.file_name_ +".csv")
        data_link = self.data_path +self.file_name +".csv"
        df_original = pd.DataFrame()
        #for i in data_file:
        df = pd.read_csv(data_link)
        df = df.dropna()
        df_original = pd.concat([df_original, df], axis=0)
        self.X_data = df_original.iloc[:, 4:].values # 5
        self.y_data = df_original.iloc[:, 2:4].values # 3 :5
        self.dim = self.X_data.shape[1]
        st.markdown(f'There are **{df_original.shape[0]}** blocks in the data')

    def __len__(self):
        return int(np.floor(self.X_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.X_indexes = np.arange(self.X_data.shape[0])
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temps = [self.X_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X_data.shape[0])
        if self.shuffle ==False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        # self._Create_large_data(self)
        X = np.empty((self.batch_size, self.dim))
        y = []
        for i, ID in enumerate(list_IDs_temps):
            X[i,] = self.X_data[ID]
            X = X.astype('float32')
            y.append(self.y_data[ID])
        X = X[:, :, np.newaxis]
        y = np.array(y, dtype=np.float32)
        return X, y


class Data_creating_final_stream_list():
    def __init__(self, data_path, link_save, file_name, ori_fs):
        self.data_path = data_path
        self.file_name = file_name
        self.link_save = link_save
        self.ori_fs = ori_fs
        self.X_index = None
        self.y_index = None

    def main(self, data_length=10, overlap=3):

        X, y = self._Final_data_cutting(data_length, overlap)
        df = self._Create_df(X, y)
        # if index % 5 == 0 and index > 0:
        #   df_original.to_csv(self.link_save + self.file_name + '.csv')
        df.to_csv(self.link_save + self.file_name +'.csv')
        return df

    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def _butter_bandpass_filter(self, data, lowcut, highcut, order):
        b, a = self._butter_bandpass(lowcut, highcut, fs=self.ori_fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _bsw_remove(self, signal, s_rate):
        signal = np.array(BaselineWanderRemoval.fix_baseline_wander(signal, s_rate))
        return signal

    def _preprocessing(self, nfs=125, hcut=0.28, lcut=25):

        data_name = self.data_path + self.file_name
        signal = wf.rdrecord(data_name)
        ecg_data = signal.p_signal[:, 0]
        # bp_data = signal.p_signal[:, -1]

        ## resample the fs:
        ecg_data_resample = decimate(ecg_data, int(self.ori_fs / nfs))
        # bp_data_resample = decimate(bp_data, int(ofs / nfs))

        ## bandpass filter and baseline removal
        ecg_data_bp = self._butter_bandpass_filter(ecg_data_resample, hcut, lcut, order=2)
        ecg_data_bp = self._bsw_remove(ecg_data_bp, nfs)

        # BP_peak, va_ = find_peaks(ecg_data_bp, height=0.5,
        # distance=50)  # try to find the peaks with the suitable distance
        return ecg_data_bp

    def _time_window(self, ecg_data, time=10, overlap_time=3, state='large', fs=125):
        len_ecg_data = ecg_data.shape[0]

        data_index = np.arange(time * fs)[None, :] + np.arange(0, len_ecg_data - time * fs, overlap_time * fs)[:, None]

        final_ecg_data = ecg_data[data_index][4:-5]

        if state == 'large':
            return final_ecg_data
        else:
            final_ecg_data = np.reshape(final_ecg_data, [final_ecg_data.shape[0], time, fs])
            final_ecg_data = np.transpose(final_ecg_data, axes=[0, 2, 1])
            return final_ecg_data  # final_bp_data

    def _simple_normalize(self, ecg_data):
        ecg_new = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
        return ecg_new

    def _Create_df(self, X, y):
        df = pd.DataFrame(X)
        df_1 = pd.DataFrame(y, columns=['SBP', 'DBP'])
        df_name = pd.DataFrame(['Sub_' + self.file_name for i in range(X.shape[0])], columns=['Name'])
        df = pd.concat([df_name, df_1, df], axis=1)
        return df

    def _Final_data_cutting(self, second, overlap):
        second = second
        overlap = overlap
        fs = 125
        num_output = 2

        leng_data = second * fs
        Acc_X = np.zeros((1, leng_data))

        # self.cutting_state = cutting_state
        ecg_data = self._preprocessing(nfs=125, hcut=0.28, lcut=25)

        # Normalize the data
        ecg_new = self._simple_normalize(ecg_data)
        # ecg_new = self._minmax_scaler(ecg_new)

        ecg_data = self._time_window(ecg_new, second, overlap, state='large', fs=125)

        Acc_X = np.append(Acc_X, ecg_data, axis=0)

        number_y_blocks = Acc_X.shape[0]
        Acc_y = np.zeros((number_y_blocks, 2))
        return Acc_X[1:], Acc_y[1:]



    #'-----------------'
#'---CREATE THE WEB API---'
## Create the front end:
link_img = 'web-logo.png'
st.image(link_img, width= 300)
st.title('BLOOD PRESSURE ESTIMATION FROM ECG LEAD II')

st.write("""
## THE DATASET IS ACCESSED FROM PHYSIONET
""")

st.write("""
### DATASET: AUTONOMIC AGING [link](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/)

""")
## side bar
mode = ['Test Old Data', 'New Data']

st.sidebar.write("""
### CHOOSE YOUR MODE:
""")

choosing_mode = st.sidebar.radio('There are two mode:',mode)

if choosing_mode == 'Test Old Data':
    ## create list subject for choosing:
    link = 'Data/Train_Val_Test_sub_data_final.pickle'
    file_to_read = open(link, "rb")
    loaded_object = pickle.load(file_to_read)
    Val_sub = loaded_object['Val_sub']
    Test_sub = loaded_object['Test_sub']

    options = st.sidebar.multiselect(
        'Choose a Subject', np.sort(Val_sub + Test_sub))

    ##CREATE THE MODEL:
    if len(options) > 0:
        model = Create_the_model(a = 0)

        link_data_csv = 'Data/'
        val_generator = DataGenerator_old_data(link_data_csv,options[0], state='Val')

        scores = model.evaluate(val_generator, verbose=0)
        st.markdown(f'The average MAE score: **{np.round (scores[1],2)}**')

        value = []
        result = []
        predict = []
        for i in range(len(val_generator)):
            a, b = val_generator.__getitem__(i)
            value.append(b)
            result.append(model.predict(a))

        final_result = np.array(value)
        final_predict = np.array(result)

        final_result = np.reshape(final_result, (final_result.shape[0]*final_result.shape[1],2))
        final_predict = np.reshape(final_predict, (final_predict.shape[0] * final_predict.shape[1], 2))
        #
        sbp_score = tf.keras.metrics.mean_absolute_error(final_result[:,0], final_predict[:,0])
        dbp_score = tf.keras.metrics.mean_absolute_error(final_result[:,1], final_predict[:,1])
        st.markdown(f'The MAE score for systolic blood pressure: **{np.round(sbp_score,2)}**')
        st.markdown(f'The MAE score for diastolic blood pressure: **{np.round(dbp_score,2)}**')


        ## draw the matrix:
        st.markdown('**Model prediction value:**')
        min_bp = np.min(final_predict, axis = 0)
        mean_bp =np.mean(final_predict,axis = 0)
        max_bp = np.max(final_predict, axis = 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min Blood Pressure",f'{int(min_bp[0])} mmHG', f'{int(min_bp[1])} mmHG')
        col2.metric("Mean Blood Pressure", f'{int(mean_bp[0])} mmHG', f'{int(mean_bp[1])} mmHG')
        col3.metric("Max Blood Pressure", f'{int(max_bp[0])} mmHG', f'{int(max_bp[1])} mmHG')

        st.write(" ")

        #Draw the data

        fig, ax = plt.subplots(figsize=(17, 10), dpi=90)
        ax.set_title('The systolic blood pressure value ')
        ax.plot(final_result[:, 0], 'o',c='orange')
        ax.plot(final_predict[:, 0], 'o',c='black')
        ax.legend(['Grouth truth','Predict'],loc = 'upper right')
        ax.set_ylabel('mmHG')
        ax.set_xlabel('Time')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(17, 10), dpi=90)
        ax.set_title('The diastolic blood pressure value ')
        ax.plot(final_result[:, 1], 'o',c='blue')
        ax.plot(final_predict[:, 1], 'o',c='black')
        ax.legend(['Grouth truth','Predict'], loc = 'upper right')
        ax.set_ylabel('mmHG')
        ax.set_xlabel('Time')
        st.pyplot(fig)

elif choosing_mode == 'New Data':
    count = 0

    ### upload the data
    # link data for saving:
    link_upload = 'Data_uploaded/'
    link_data_save = 'Data_procceed_after_upload/'

    # display the sidebar
    st.sidebar.markdown(f'**Upload 2 files: .dat, .head**')
    ori_fs = st.sidebar.number_input('Insert the original frequency')

    uploaded_files = st.sidebar.file_uploader("", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if count < 2:
            with open(os.path.join(link_upload, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                count+=1
                data_file_name = uploaded_file.name[:-4]



    #data_file_name = data_file_name
    if len(uploaded_files) == 2:
        a = Data_creating_final_stream_list(link_upload, link_data_save, data_file_name, ori_fs)
        df = a.main()


        ### call the model and predict the data:

        model = Create_the_model(a = 0)

        link_data_csv = 'Data_procceed_after_upload/'
        name_sub = 'Sub_' + data_file_name
        val_generator = DataGenerator_new_data(link_data_csv,data_file_name, state='Val')


        result = []
        for i in range(len(val_generator)):
            a, b = val_generator.__getitem__(i)
            result.append(model.predict(a))

        final_predict = np.array(result)

        final_predict = np.reshape(final_predict, (final_predict.shape[0] * final_predict.shape[1], 2))

        ## draw the matrix:

        min_bp = np.min(final_predict, axis = 0)
        mean_bp =np.mean(final_predict,axis = 0)
        max_bp = np.max(final_predict, axis = 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min Blood Pressure",f'{int(min_bp[0])} mmHG', f'{int(min_bp[1])} mmHG')
        col2.metric("Mean Blood Pressure", f'{int(mean_bp[0])} mmHG', f'{int(mean_bp[1])} mmHG')
        col3.metric("Max Blood Pressure", f'{int(max_bp[0])} mmHG', f'{int(max_bp[1])} mmHG')

        st.write(" ")


        ## Plot the chart
        fig, ax = plt.subplots(figsize=(17, 10), dpi=90)
        ax.set_title('The systolic blood pressure value ')
        ax.plot(final_predict[:, 0], 'o',c='orange')
        ax.legend(['Predict'],loc = 'upper right')
        ax.set_ylabel('mmHG')
        ax.set_xlabel('Time')
        st.pyplot(fig)

        st.write(" ")
        fig, ax = plt.subplots(figsize=(17, 10), dpi=90)
        ax.set_title('The diastolic blood pressure value ')
        ax.plot(final_predict[:, 1], 'o',c='blue')
        ax.legend(['Predict'], loc = 'upper right')
        ax.set_ylabel('mmHG')
        ax.set_xlabel('Time')
        st.pyplot(fig)









