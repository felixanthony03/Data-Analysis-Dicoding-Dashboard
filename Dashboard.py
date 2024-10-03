import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import statsmodels.api as sm

#Title
st.title('Bike Sharing Dashboard')
st.write('by: Felix Anthony')

#Load data day.csv
df_day= pd.read_csv('data/day.csv')

#Load data hour.csv
df_hour = pd.read_csv('data/hour.csv')

#convert to datetime object
df_day['dteday'] = pd.to_datetime(df_day['dteday'])
df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])

# Gambaran Umum Dataset
st.write('### Gambaran Umum Dataset')
st.write("""
sumber: [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- Berisi jumlah penyewaan sepeda per jam dan harian (2011-2012)
- Mencakup informasi cuaca dan musim
""")

# Create two columns
col1, col2 = st.columns(2)

# Display hour_df in the first column
with col1:
    st.write("#### Hourly Data")
    st.dataframe(df_hour, height=200)

# Display day_df in the second column
with col2:
    st.write("#### Daily Data")
    st.dataframe(df_day, height=200)

# Tentang Sistem Berbagi Sepeda
st.write('### Tentang Sistem Berbagi Sepeda')
st.write("""
- Sistem penyewaan dan pengembalian sepeda otomatis
- Pengguna bisa menyewa dari satu stasiun dan mengembalikan di stasiun lain
- Lebih dari 500 program berbagi sepeda di seluruh dunia
- Penting untuk lalu lintas, lingkungan, dan manfaat kesehatan
""")

#Analisis Lanjutan
st.header('Analisis dan Visualisasi Data')
# adding smoothing (moving average) on the cnt data
df_day['cnt_avg_14'] = df_day['cnt'].rolling(window=7, center=True).mean()
# rental bikes per day from 2011 to 2012
plt.figure(figsize=(12, 4))
plt.plot(df_day['dteday'], df_day['cnt_avg_14'], linestyle='-', color='blue')

plt.title('Rental Bikes Over the Years', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
st.pyplot(plt)
st.caption("Pengunaan Rental Bikes per hari di tahun 2011-2012")
st.write('''
Pada bagian ini akan dilakukan **analisa** dan **visualisasi** data penguanaan rental bikes dari tahun 2011-2012 berdasarkan empat kategori berikut:
''')


# Create four tabs
tab1, tab2, tab3, tab4 = st.tabs(["Months & Season", "Time of the Day", "Temperature", "Weather"])

# Content for Tab 1: Months & Season
with tab1:
    st.header("Months")
    # rental bikes based on month of the year visualization
    hours_registered = list(df_hour.groupby("mnth")["registered"].mean())
    hours_casual = list(df_hour.groupby("mnth")["casual"].mean())
    hours = list(df_hour.groupby("mnth")["cnt"].mean())


    plt.figure(figsize=(9, 3))
    intensity = (np.ones(len(hours))*0.6)
    bar_colors_registered = cm.Greens(intensity) 
    bar_colors_casual = cm.Blues(intensity)
    plt.bar(range(1,len(hours_casual)+1), hours_casual, label='Casual', color=bar_colors_casual)
    plt.bar(range(1,len(hours_registered)+1), hours_registered, label='Registered', bottom=hours_casual, color=bar_colors_registered)
    plt.xticks(range(1,len(hours_casual)+1))
    plt.xlabel("Month")
    plt.ylabel("Total Bike Rentals")
    plt.legend()
    plt.title("Total Bike Rentals per Hour")

    st.pyplot(plt)
    st.caption("Rata-rata pengunaan rental bikes per bulan di tahun 2011-2012")

    # rental bikes based on season of the year visualization

    st.header("Seasons")

    sns.set(style="whitegrid")

    plt.figure(figsize=(9,4))
    box_plot = sns.boxplot(x='season', y='cnt', data=df_day, palette="Set2", width=0.2)

    plt.title('Data Distribution \nby Season', fontsize=12)
    plt.ylabel('Rental Bikes Count', fontsize=14)
    xticks_labels = ['Spring', 'Summer', 'fall', 'winter']
    plt.xticks(ticks=range(len(xticks_labels)), labels=xticks_labels,fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    st.pyplot(plt)
    st.caption("Pengunaan rental bikes berdasarkan musim di tahun 2011-2012")

    
    st.write('''
### Insights
- Puncak aktivitas dari pengunaan rental bikes terjadi pada pertengahan tahun
- Puncah aktivitas pada musim summer dan fall
- Dapat disebabkan oleh kondisi cuaca yang lebih nyaman untuk beraktivas di luar ruangan.
''')

# Content for Tab 2: Time of the Day
with tab2:
    st.header("Time of the Day")
    # rental bikes based on time of the day visualization
    hours_registered = list(df_hour.groupby("hr")["registered"].mean())
    hours_casual = list(df_hour.groupby("hr")["casual"].mean())
    hours = list(df_hour.groupby("hr")["cnt"].mean())

    plt.figure(figsize=(10, 3.5))
    # intensity = (np.array(hours)+100) / (max(hours)+100)
    intensity = (np.ones(len(hours))*0.6)
    bar_colors_registered = cm.Greens(intensity) 
    bar_colors_casual = cm.Blues(intensity)
    plt.bar(range(1,24+1), hours_casual, label='Casual', color=bar_colors_casual)
    plt.bar(range(1,24+1), hours_registered, bottom=hours_casual, label='Registered', color=bar_colors_registered)
    plt.xticks(range(1,len(hours_casual)+1))
    plt.xlabel("Hour")
    plt.ylabel("Bike Rentals")
    plt.legend()
    plt.title("Bike Rentals per Hour")
    plt.show()

    st.pyplot(plt)
    st.caption("Rata-rata pengunaan rental bikes per jam di tahun 2011-2012")

    st.write('''
### Insights
- Puncak aktivitas pengguna sepeda berada pada jam 18:00
- Aktivitas pengguna sepeda fluktuatif, mempunyai dua puncak lokal
- Aktivitas pengguna didominasi dengan pengguana registered          
''')
    
# Content for Tab 3: Temperature
with tab3:
    st.header("Temperature")

    # Scale the temperature data
    df_day['temp_scaled'] = df_day['temp'] * 41

    # Fit the model using the original temperature data (not scaled)
    X = sm.add_constant(df_day['temp'])
    Y = df_day['cnt']
    model = sm.OLS(Y, X).fit()

    # Predict the count using the model
    df_day['predicted_cnt'] = model.predict(X)

    # Create scatter plot with scaled temperature on the x-axis
    plt.figure(figsize=(10, 6))
    plt.scatter(df_day['temp_scaled'], df_day['cnt'], color='blue', label='Data Points', alpha=0.7)

    # Plot regression line with scaled temperature on the x-axis
    plt.plot(df_day['temp_scaled'], df_day['predicted_cnt'], color='red', label='Regression Line')
     
    # Customize the plot
    plt.title('Scatter Plot with Regression Line between Temperature and Rental Bikes', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature in Celcius', fontsize=14)
    plt.ylabel('Rental Bikes', fontsize=14)
    plt.legend()
    plt.grid()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # Show the plot
    plt.tight_layout()
    plt.show()


    st.pyplot(plt)
    st.caption("Pengunaan rental bikes berdasarkan temperature di tahun 2011-2012")

    # categorize the temperature based on quartil
    df_day['temp_category'] = pd.qcut(df_day['temp'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    # rental bikes based on temp categories

    sns.set(style="whitegrid")

    plt.figure(figsize=(10,5))
    box_plot = sns.boxplot(x='temp_category', y='cnt', data=df_day, palette="Set2", width=0.2)

    plt.title('Data Distribution \nby Temperature', fontsize=12)
    plt.ylabel('Rental Bikes Count', fontsize=14)
    plt.xlabel('Temperature', fontsize=14)

    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    st.pyplot(plt)
    st.caption("Box plot pengunaan rental bikes berdasarkan temperature yang dibagi menjadi 4 kuartil")

    st.write('''
### Insights
- Temperatur dan rental bikes mempunyai korelasi positif
- Semakin hangat tempertur semakin banyak rental bikes
- Dapat disebabkan oleh kondisi suhu yang lebih nyaman untuk beraktivas di luar ruangan.       
''')


# Content for Tab 4: Weather
with tab4:
    st.header("Weather")
    st.write('''
#### Weather Conditions:

1. Clear, Few clouds, Partly cloudy, Partly cloudy
2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3. Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4. Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
''')
    # rental bikes based on weather visualization
    hours_registered = list(df_hour.groupby("weathersit")["registered"].mean())
    hours_casual = list(df_hour.groupby("weathersit")["casual"].mean())
    hours = list(df_hour.groupby("weathersit")["cnt"].mean())


    plt.figure(figsize=(10, 6))
    intensity = (np.ones(len(hours))*0.6)
    bar_colors_registered = cm.Greens(intensity)  # Create a gradient based on the values
    bar_colors_casual = cm.Blues(intensity)
    plt.bar(range(1,len(hours_casual)+1), hours_casual, label='Causal', color=bar_colors_casual)
    plt.bar(range(1,len(hours_registered)+1), hours_registered, label='Registered', bottom=hours_casual, color=bar_colors_registered)
    plt.xticks(range(1,len(hours_casual)+1))
    plt.xlabel("Weather condition")
    plt.ylabel("Bike Rentals")
    plt.legend()
    plt.title("Bike Rentals by Weather Condition")
    plt.show()

    st.pyplot(plt)
    st.caption("Pengunaan rental bikes berdasarkan kondisi cuaca di tahun 2011-2012")

    st.write('''
### Insights
- Aktivitas penggunaan rental bikes dipengengaruhi oleh kondisi cuaca
- Aktivitas penggunaan rental bikes terbanyak ketika kondisi cuaca 1
- Aktivitas pengguna terendah ketika kondisi cuaca 4
- Dapat disebabkan oleh kondisi cuaca yang lebih nyaman untuk beraktivas di luar ruangan.    
''')


