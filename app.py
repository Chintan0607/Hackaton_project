from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask('Tech_titans')

# Load the trained machine learning model
# model = joblib.load('your_model_file.pkl')  # Update with your model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        # temperature = float(request.form['temperature'])
        # humidity = float(request.form['humidity'])
        # occupancy = int(request.form['occupancy'])
        
        import pandas as pd
        import numpy as np
        import math
        import random
        import matplotlib.pyplot as plt
        from tensorflow.keras.models import load_model
        import seaborn as sns
        from datetime import datetime
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score

        # Assuming df is your DataFrame with a 'Timestamp' column
        df = pd.read_csv(r'modified_dataset.csv')
        import pandas as pd
        from datetime import datetime

        # Function to convert timestamp to separate columns
        def convert_timestamp(timestamp):
            dt_object = datetime.strptime(timestamp, '%d-%m-%Y %H.%M')
            day = dt_object.day
            month = dt_object.month
            year = dt_object.year
            hour = dt_object.hour
            minute = dt_object.minute

            # Determine the time of the day
            if 6 <= hour < 12:
                time_of_day = 'Morning'
            elif 12 <= hour < 18:
                time_of_day = 'Afternoon'
            elif 18 <= hour < 24:
                time_of_day = 'Evening'
            else:
                time_of_day = 'Night'

            return day, month, year, hour, minute, time_of_day

        # Apply the function to the timestamp column
        df[['Day', 'Month', 'Year', 'Hour', 'Minute', 'Time_of_day']] = df['ts'].apply(lambda x: pd.Series(convert_timestamp(x)))

        # Save the modified DataFrame to a new CSV file
        df.to_csv('output.csv', index=False)
        time_of_day_mapping = {'Night': 0, 'Evening': 1, 'Morning': 2, 'Afternoon': 3}

        # Map time of day values to integers and create a new column
        df['Time_of_day_int'] = df['Time_of_day'].map(time_of_day_mapping)

        # Drop the original column
        df.drop(columns=['Time_of_day'], inplace=True)

        # Save the modified DataFrame to a new CSV file
        df.to_csv('output.csv', index=False)
        print(df['Time_of_day_int'].unique())

        print(df.info())

        humidity_columns = [f'humidity{i}' for i in range(1, 9)]
        temperature_columns = [f'temperature{i}' for i in range(1, 9)]

        # Calculate the average humidity and temperature
        average_humidity = df[humidity_columns].mean(axis=1)
        average_temperature = df[temperature_columns].mean(axis=1)

        df.to_csv('output_file.csv', index=False)

        # Add the average values to the DataFrame
        df['average_humidity'] = average_humidity
        df['average_temperature'] = average_temperature

        print(df.info())
        print(df.head(5).to_string())

        df['Occupancy'] = np.random.randint(0, 10, size=len(df))
        df.to_csv('modified_dataset.csv', index=False)

        #adjust_ac_temperature
        def calculate_pmv(temperature, humidity, met, clo):
            """Calculate Predicted Mean Vote (PMV) index."""
            # Constants for the PMV calculation
            tr = temperature  # Mean radiant temperature in Celsius
            vel = 0.1  # Air velocity in m/s (assuming a typical indoor environment)
            rh = humidity  # Relative humidity in %

            # Constants for metabolic rate (met) adjustment
            met_factor = {
                0.7: 1.2,
                1.0: 1.0,
                1.2: 0.8,
                1.5: 0.6,
                2.0: 0.5,
                2.5: 0.4
            }

            # Constants for clothing insulation (clo) adjustment
            clo_factor = {
                0.5: 0.5,
                1.0: 1.0,
                1.5: 1.5,
                2.0: 2.0,
                3.0: 2.0
            }

            # Adjust metabolic rate (met) and clothing insulation (clo) based on specified values
            M = met * 58.15 * met_factor[met]
            Icl = clo * clo_factor[clo]

            # Heat transfer coefficient for forced convection in W/m^2K
            hc = 12.1 * (vel / 0.155) ** 0.5 if vel > 0.155 else 2.38 * (vel ** 0.5)

            # Air temperature of the room in Celsius
            tr_k = tr + 273.15

            # Sensible heat loss or gain in W/m^2
            h_sens = 3.96 * 10 ** -8 * (0.0000000567 * (tr_k ** 4 - tr_k ** 4) + 0.0038 * (tr_k - tr_k))

            # Mean radiant temperature in Celsius
            t_mrt = tr

            # Skin temperature in Celsius
            t_skin = 35.7 - 0.028 * (M - 58.15 * met_factor[met]) - 0.155 * (temperature - tr)

            # PMV calculation
            pmv = (0.303 * math.exp(-0.036 * M) + 0.028) * (
                        M - 58.15 * met_factor[met] - h_sens - 0.00305 * (5733 - 6.99 * (M - 58.15 * met_factor[met]) - rh)
                        - Icl * (3.96 * 10 ** -8 * (t_skin ** 4 - t_mrt ** 4) + hc * (t_skin - t_mrt)))

            return pmv

        
        def adjust_ac_temperature(temperature, humidity, occupancy):
            """Adjust AC temperature based on thermal comfort and occupancy."""
            # Randomly generate metabolic rate (met) and clothing insulation (clo)
            met = random.choice([0.7, 1.0, 1.2, 1.5, 2.0, 2.5])  # Typical metabolic rates (met) in met
            clo = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])  # Typical clothing insulation (clo) in clo

            # Calculate PMV
            pmv = calculate_pmv(temperature, humidity, met, clo)

            # Adjust AC temperature based on PMV and occupancy
            if pmv < -0.5:
                return temperature + 1 + (0.1 * occupancy)
            elif pmv > 0.5:
                return temperature - 1 - (0.1 * occupancy)
            else:
                return temperature


        # def adjust_ac_temperature(temperature, humidity):
        #     """Adjust AC temperature based on thermal comfort."""
        #     # Randomly generate metabolic rate (met) and clothing insulation (clo)
        #     met = random.choice([0.7, 1.0, 1.2, 1.5, 2.0, 2.5])  # Typical metabolic rates (met) in met
        #     clo = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])  # Typical clothing insulation (clo) in clo

        #     # Calculate PMV
        #     pmv = calculate_pmv(temperature, humidity, met, clo)

        #     # Adjust AC temperature based on PMV
        #     if pmv < -0.5:
        #         return temperature + 1
        #     elif pmv > 0.5:
        #         return temperature - 1
        #     else:
        #         return temperature

        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        occupancy = int(request.form['occupancy'])
        df['AC_Temperature'] = df.apply(lambda row: adjust_ac_temperature(temperature, humidity, occupancy), axis=1)
        df.to_csv('modified_dataset.csv', index=False)
        print(df.info())
        print(df.head(10).to_string())

        '''plt.figure(figsize=(8, 6))
        sns.histplot(df['AC_Temperature'], bins=20, kde=True, color='skyblue')
        plt.title('Histogram of AC Temperature')
        plt.xlabel('AC Temperature')
        plt.ylabel('Frequency')

        plt.figure(figsize=(12, 6))
        sns.lineplot(x='ts', y='AC_Temperature', data=df, color='green')
        plt.title('AC Temperature Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('AC Temperature')
        plt.xticks(rotation=45)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Time_of_day_int', y='AC_Temperature', data=df)
        plt.title('AC Temperature by Time of Day')
        plt.xlabel('Time of Day')
        plt.ylabel('AC Temperature')
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Night', 'Evening', 'Morning', 'Afternoon'])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='average_humidity', y='AC_Temperature', data=df, color='red')
        plt.title('AC Temperature vs. Humidity')
        plt.xlabel('Average Humidity')
        plt.ylabel('AC Temperature')

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Time_of_day_int', y='AC_Temperature', data=df)
        plt.title('AC Temperature by Time of Day (Violin Plot)')
        plt.xlabel('Time of Day')
        plt.ylabel('AC Temperature')
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Night', 'Evening', 'Morning', 'Afternoon'])

        sns.pairplot(df[['average_humidity', 'average_temperature', 'AC_Temperature']])

        plt.figure(figsize=(10, 8))
        df1 = df.drop(['ts'],axis=1)
        sns.heatmap(df1.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')'''

        # Encode time_of_day_int using one-hot encoding
        df = pd.get_dummies(df, columns=['Time_of_day_int'])

        # Split into input (X) and output (y) variables
        df.drop(['ts'],axis=1,inplace=True)
        X = df.drop(['AC_Temperature'],axis=1)
        y = df[['AC_Temperature']]

        # Scale the input features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Reshape the input data for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Define the LSTM model
        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=X_train.shape[2]))  # Output units match the number of features in X_train

        # Compile the model
        model.compile(optimizer='adam', loss='mse')


        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print("Test Loss:", loss)

        # Predictions on the test set
        predictions = model.predict(X_test)

        # Inverse scaling on the predictions
        predictions = scaler.inverse_transform(predictions)

        # Print the predicted AC temperature
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        # Print the predicted AC temperature
        print("Predicted AC Temperature:")
        print(predictions)

        # Calculate the average predicted AC temperature
        average_predicted_temperature = np.mean(predictions)

        '''plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test['AC_Temperature'], label='Actual AC Temperature', color='blue')
        plt.plot(y_test.index, predictions, label='Predicted AC Temperature', color='red', linestyle='--')
        plt.title('Actual vs. Predicted AC Temperature')
        plt.xlabel('Index')
        plt.ylabel('AC Temperature')
        plt.legend()'''

        # Print the average predicted AC temperature
        print("Average Predicted AC Temperature:", average_predicted_temperature)
        # print(average_predicted_temperature.size())
        ans = []
        for ele in predictions:
            print(ele[-1])
            ans.append(ele[-1])
        '''
        y_test_reshaped = y_test.values.reshape(predictions.shape)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, predictions)
        print("Mean Absolute Error:", mae)

        mse = mean_squared_error(y_test,predictions)
        print("Mean Squared Error: ",mse)

        rmse = np.sqrt(mse)
        print("Root Mean Squared Error: ",rmse)

        r2 = r2_score(y_test, predictions)
        print("R-squared (R^2):", r2)

        plt.show()'''

        # print(temperature, humidity, occupancy)

        # Make prediction

        # prediction = model.predict([[temperature, humidity, occupancy]])

        # Save prediction to a file
        # with open('prediction.txt', 'w') as file:
        #     file.write(str(prediction[0]))

        return render_template('result.html', ans = ans)

if __name__ == '__main__':
    app.run(debug=True)
