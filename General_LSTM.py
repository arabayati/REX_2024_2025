
cluster = 8
# Libraries
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import scipy.stats
from keras import optimizers
from datetime import datetime

# Parameters
TargetLabel = 'streamflow_mmd'
LearningRate = 0.001
TIME_STEP = 365
EPOCHs = 40
BatchSize = 200
Patience = 15
TrainRatio = 0.4
ValidationRatio = 0.2
TestRatio = 0.4

# Input columns
f_columns = ['mean_temperature_C', 'precipitation_mmd', 'pet_mmd']
staticColumns = ['area_km2', 'mean_elevation_m', 'mean_slope_mkm', 'shallow_soil_hydc_md', 'soil_hydc_md', 'soil_porosity', 'depth_to_bedrock_m', 'maximum_water_content_m', 'bedrock_hydc_md', 'soil_bedrock_hydc_ratio', 'mean_precipitation_mmd', 'mean_pet_mmd', 'aridity', 'snow_fraction', 'seasonality', 'high_P_freq_daysyear', 'low_P_freq_daysyear', 'high_P_dur_day', 'low_P_dur_day', 'mean_forest_fraction_percent']

print(len(staticColumns))

# Input folder (daily csv files)
folder = f'/home/majidara/Data_Daily_Clustered_based_on_AI_SF_SI/Cluster_{cluster}/'

# Output folder, where we save the results
outputfolder = f'/home/majidara/General_LSTM_weights/{cluster}/'

if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)
    print('Oops! directory did not exist, but no worries, I created it!')

SaveModel = outputfolder

# Static Data - it must contain items listed by "staticColumns" and grid code
path_static = '/home/majidara/stats/attributes.csv'

# Read and Normalize statistical features
dfs = pd.read_csv(path_static)
OurDesiredStaticAttributes = staticColumns
f_transformer = StandardScaler().fit(dfs[OurDesiredStaticAttributes])
dfs[OurDesiredStaticAttributes] = f_transformer.transform(dfs[OurDesiredStaticAttributes])
dfs['gridcode'] = pd.read_csv(path_static)['gridcode']

# Create Dataset function
def create_dataset(X, y, date_df, doy_df, time_steps=1):
    Xs, ys, date, doy = [], [], [], []
    for i in range(len(X) - time_steps):
        X_seq = X.iloc[i:(i + time_steps)]

        if not X_seq.isnull().values.any() and not pd.isnull(y.iloc[i + time_steps-1]):
            Xs.append(X_seq.values)
            ys.append(y.iloc[i + time_steps-1])
            date.append(date_df.iloc[i + time_steps-1])
            doy.append(doy_df.iloc[i + time_steps-1])

    return np.array(Xs), np.array(ys), np.array(date), np.array(doy)

# NSE function (Nash-Sutcliff-Efficiency)
def NSE(targets, predictions):
    return 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

# Model definition
model = keras.Sequential()
model.add(keras.layers.LSTM(units=256, return_sequences=False, input_shape=(TIME_STEP, 23)))
model.add(keras.layers.Dropout(rate=0.4))
model.add(keras.layers.Dense(units=1))

callbacks = [keras.callbacks.EarlyStopping(patience=Patience, restore_best_weights=True)]
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LearningRate))



# Check if any .h5 file exists in the output directory
h5_files = glob.glob(os.path.join(outputfolder, '*.h5'))

if h5_files:
    # If any .h5 file is found, load the weights from the latest one
    latest_model = max(h5_files, key=os.path.getctime)  # Get the latest .h5 file by creation time
    print(f"Loading weights from {latest_model}")
    model.load_weights(latest_model)
else:
    # If no .h5 file found, continue without loading
    print("No previous model weights found, starting from scratch.")





TraindGridCodes = np.array([])

for file in os.listdir(folder):
    if file.endswith(".csv"):
        GridCode = int(file.rstrip(".csv"))
        print('GridCode: ', GridCode)

        Dir = os.path.join(folder, file)
        df = pd.read_csv(Dir)
        df['date'] = pd.to_datetime(df.pop('date'))
        df['day_of_year'] = df['date'].dt.dayofyear

        # Chronological splitting of data (Train 0.4, Val 0.2, Test 0.4)
        train_size = int(len(df) * TrainRatio)
        val_size = int(len(df) * ValidationRatio)
        test_size = int(len(df) * TestRatio)

        train, val, test = df.iloc[:train_size].copy(), df.iloc[train_size:train_size + val_size].copy(), df.iloc[train_size + val_size:].copy()

        # Normalize Input Data
        f_transformer = StandardScaler().fit(train[f_columns])
        train[f_columns] = f_transformer.transform(train[f_columns])
        val[f_columns] = f_transformer.transform(val[f_columns])
        test[f_columns] = f_transformer.transform(test[f_columns])

        # Apply log transformation to the target (log(x+1))
        train[TargetLabel] = np.log1p(train[TargetLabel])
        val[TargetLabel] = np.log1p(val[TargetLabel])
        test[TargetLabel] = np.log1p(test[TargetLabel])

        # Add static data for each row
        static_row = dfs[dfs['gridcode'] == GridCode]
        for item in staticColumns:
            train.loc[:, item] = static_row[item].values[0]
            val.loc[:, item] = static_row[item].values[0]
            test.loc[:, item] = static_row[item].values[0]

        input_columns = f_columns + staticColumns

        # Create datasets
        X_train, y_train, train_date, train_days = create_dataset(train[input_columns], train[TargetLabel], train['date'], train['day_of_year'], time_steps=TIME_STEP)
        X_val, y_val, val_date, val_days = create_dataset(val[input_columns], val[TargetLabel], val['date'], val['day_of_year'], time_steps=TIME_STEP)
        X_test, y_test, test_date, test_days = create_dataset(test[input_columns], test[TargetLabel], test['date'], test['day_of_year'], time_steps=TIME_STEP)

        # Fit the model
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHs,
            batch_size=BatchSize,
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=callbacks
        )

        # Predict on the test set
        y_pred_test = model.predict(X_test,batch_size = 500)

        # Inverse log transformation to get back the original scale
        y_test_orig = np.expm1(y_test)
        y_pred_test_orig = np.expm1(y_pred_test)

        # Calculate NSE for test set
        nse_test = NSE(y_test_orig, y_pred_test_orig.flatten())
        print(f"Test NSE for GridCode {GridCode}: {nse_test}")

        # Save progress
        TraindGridCodes = np.append(TraindGridCodes, GridCode)
        np.savetxt(SaveModel + 'Grodcodes_Based_On_Which_Trained_sofar.out', TraindGridCodes, delimiter=',')
        model.save_weights(SaveModel + 'Generally_Trained_UP_TO_NOW_Model.h5')

        os.remove(Dir)

# Final model save
model.save_weights(SaveModel + 'Generally_Trained_Model.h5')
