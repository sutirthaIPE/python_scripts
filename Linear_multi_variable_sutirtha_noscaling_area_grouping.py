#created this version to save the code for github for version control
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# --------Generate some sample data to work with --------------
df = pd.read_csv("eRTD_TD_Delta_Summary_table_polished.csv")
df2 = df[df['Socket']!='Split']
X1 = df2['Area']
X2 = df2['Set point']
Y = df2['Delta_SOT']

# -- generate a dataset (PD dataframe) to store the info --
df3 = pd.DataFrame({"X1":X1,"X2":X2,"Y":Y})

# -- Generating 4 sets of dataframe based on the area -------------
''' I am doing it with hard coding, later this sould be implemented using Loop option '''

# area will be grouped as below
#Here areas are created based on the 4 different range

# area 1 : 0 < area <= 45
New_df_area1 = df3[(df3['X1'] > 0) & (df3['X1']<=45)]
# area 2 : 45 < area <= 100
New_df_area2 = df3[(df3['X1'] > 45) & (df3['X1']<=100)]
# area 3 : 100 < area <= 200
New_df_area3 = df3[(df3['X1'] > 100) & (df3['X1']<=200)]
# area 4 : arae > 200
New_df_area4 = df3[(df3['X1'] > 200)]


# --- SPEC values of slope and intercept
# -- area 1 ---------
m1 = 0.03
b1 = -0.76
# ---- area 2 -----
m2 = 0.03
b2 = -0.76
# ---- area 3 -----
m3 = 0.03
b3 = -0.76
# ---- area 4 -----
m4 = 0.03
b4 = -0.76


#---------- Linear regression for area 1 (0-45 mm^2) -------------
arr1 = New_df_area1["X2"].values
arr2 = New_df_area1["Y"].values
re_arr1 = arr1.reshape(-1,1)

# Defining the training and the test data for area1

X1, y1 = re_arr1, arr2
X_train_area1, X_test_area1, y_train_area1, y_test_area1 = train_test_split(X1, y1, test_size=0.15, random_state=42)
lin_reg_model_area1 = LinearRegression()
lin_reg_model_area1.fit(X_train_area1, y_train_area1)
lin_reg_y_predicted_area1 = lin_reg_model_area1.predict(X_test_area1)
lin_reg_rmse_area1 = mean_squared_error(y_test_area1, lin_reg_y_predicted_area1)
r2_area1 = r2_score(y_test_area1, lin_reg_y_predicted_area1)
lin_reg_y_train_area1 = lin_reg_model_area1.predict(X_train_area1)
lin_reg_y_total_area1 = lin_reg_model_area1.predict(X1)

print("coeffecient, intercept, RMSE of the model and R**2 values for each area")
print("==========================================================")
print()
print("A)" + " " + "area 1 = 0 < area <= 45")
print('─' * 80)
print("coefficient_area1:", lin_reg_model_area1.coef_,"intercept_area1:", lin_reg_model_area1.intercept_)
print("RMSE_area1:",lin_reg_rmse_area1)
print(f"R**2 value area1: {r2_area1}")

#---------- Linear regression for area 2 (45-100 mm^2)-------------
arr1  = New_df_area2["X2"].values
arr2 = New_df_area2["Y"].values
re_arr1 = arr1.reshape(-1,1)

# Defining the training and the test data for area2

X2, y2 = re_arr1, arr2
X_train_area2, X_test_area2, y_train_area2, y_test_area2 = train_test_split(X2, y2, test_size=0.15, random_state=42)
lin_reg_model_area2 = LinearRegression()
lin_reg_model_area2.fit(X_train_area2, y_train_area2)
lin_reg_y_predicted_area2 = lin_reg_model_area2.predict(X_test_area2)
lin_reg_rmse_area2 = mean_squared_error(y_test_area2, lin_reg_y_predicted_area2)
r2_area2 = r2_score(y_test_area2, lin_reg_y_predicted_area2)
lin_reg_y_train_area2 = lin_reg_model_area2.predict(X_train_area2)
lin_reg_y_total_area2 = lin_reg_model_area2.predict(X2)


print("==========================================================")
print()
print("B)" + " " + "area 2 = 45 < area <= 100")
print('─' * 80)
print("coefficient_area2:", lin_reg_model_area2.coef_,"intercept_area2:", lin_reg_model_area2.intercept_)
print("RMSE_area2:",lin_reg_rmse_area2)
print(f"R**2 value area2: {r2_area2}")

#---------- Linear regression for area 3 (100-200 mm^2)-------------
arr1  = New_df_area3["X2"].values
arr2 = New_df_area3["Y"].values
re_arr1 = arr1.reshape(-1,1)

# Defining the training and the test data for area3
X3, y3 = re_arr1, arr2
X_train_area3, X_test_area3, y_train_area3, y_test_area3 = train_test_split(X3, y3, test_size=0.15, random_state=42)
lin_reg_model_area3 = LinearRegression()
lin_reg_model_area3.fit(X_train_area3, y_train_area3)
lin_reg_y_predicted_area3 = lin_reg_model_area3.predict(X_test_area3)
lin_reg_rmse_area3 = mean_squared_error(y_test_area3, lin_reg_y_predicted_area3)
r2_area3 = r2_score(y_test_area3, lin_reg_y_predicted_area3)
lin_reg_y_train_area3 = lin_reg_model_area3.predict(X_train_area3)
lin_reg_y_total_area3 = lin_reg_model_area3.predict(X3)

print("==========================================================")
print()
print("C)" + " " + "area 3 = 100 < area <= 200")
print('─' * 80)
print("coefficient_area3:", lin_reg_model_area3.coef_,"intercept_area3:", lin_reg_model_area3.intercept_)
print("RMSE_area3:",lin_reg_rmse_area3)
print(f"R**2 value area3: {r2_area3}")

#---------- Linear regression for area 4 (200 + mm^2)-------------
arr1  = New_df_area4["X2"].values
arr2 = New_df_area4["Y"].values
re_arr1 = arr1.reshape(-1,1)

# Defining the training and the test data for area4

X4, y4 = re_arr1, arr2
X_train_area4, X_test_area4, y_train_area4, y_test_area4 = train_test_split(X4, y4, test_size=0.15, random_state=42)
lin_reg_model_area4 = LinearRegression()
lin_reg_model_area4.fit(X_train_area4, y_train_area4)
lin_reg_y_predicted_area4 = lin_reg_model_area4.predict(X_test_area4)
lin_reg_rmse_area4 = mean_squared_error(y_test_area4, lin_reg_y_predicted_area4)
r2_area4 = r2_score(y_test_area4, lin_reg_y_predicted_area4)
lin_reg_y_train_area4 = lin_reg_model_area4.predict(X_train_area4)
lin_reg_y_total_area4 = lin_reg_model_area4.predict(X4)

print("==========================================================")
print()
print("D)" + " " + "area 4 = 200 < area")
print('─' * 80)
print("coefficient_area4:", lin_reg_model_area4.coef_,"intercept_area4:", lin_reg_model_area4.intercept_)
print("RMSE_area4:",lin_reg_rmse_area4)
print(f"R**2 value area4: {r2_area4}")


# ------ printing results in a CSV file ---------
df4 = pd.DataFrame({"Area(mm^2)":["0-45","45-100","100-200","200+"],
                    "Slopes" : [lin_reg_model_area1.coef_, lin_reg_model_area2.coef_, lin_reg_model_area3.coef_, lin_reg_model_area4.coef_],
                    "Intercept" : [lin_reg_model_area1.intercept_, lin_reg_model_area2.intercept_, lin_reg_model_area3.intercept_, lin_reg_model_area4.intercept_],
                    "RMSE" : [lin_reg_rmse_area1,lin_reg_rmse_area2,lin_reg_rmse_area3,lin_reg_rmse_area4],
                    "R^2 value" : [r2_area1,r2_area2,r2_area3,r2_area4]})

df4.to_csv('ML_regression_results_all.csv',index=True)

# Plot the results (visualization of Linear regression model vs actual data and the SPEC results for the whole Datasets)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)


axes[0,0].scatter(X1, y1,color='blue', label='Actual data', s=100, alpha=0.5)
axes[0,0].scatter(X1, lin_reg_y_total_area1, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,0].plot(X1, lin_reg_y_total_area1)
axes[0,0].plot(X1, m1*X1 + b1, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,0].set_xlabel('Set_point')
axes[0,0].set_ylabel('Delta_SOT')
axes[0,0].set_title('Linear Regression: Set point vs area (area = 0-45 mm^2)')
axes[0,0].grid()
axes[0,0].legend()

axes[0,1].scatter(X2, y2,color='blue', label='Actual data', s=100, alpha=0.5)
axes[0,1].scatter(X2, lin_reg_y_total_area2, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,1].plot(X2, lin_reg_y_total_area2)
axes[0,1].plot(X2, m2*X2 + b2, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,1].set_xlabel('Set_point')
axes[0,1].set_ylabel('Delta_SOT')
axes[0,1].set_title('Linear Regression: Set point vs area (area = 45-100 mm^2)')
axes[0,1].grid()
axes[0,1].legend()

axes[1,0].scatter(X3, y3,color='blue', label='Actual data', s=100, alpha=0.5)
axes[1,0].scatter(X3, lin_reg_y_total_area3, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,0].plot(X3, lin_reg_y_total_area3)
axes[1,0].plot(X3, m3*X3 + b3, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,0].set_xlabel('Set_point')
axes[1,0].set_ylabel('Delta_SOT')
axes[1,0].set_title('Linear Regression: Set point vs area (area = 100-200 mm^2)')
axes[1,0].grid()
axes[1,0].legend()

axes[1,1].scatter(X4, y4,color='blue', label='Actual data', s=100, alpha=0.5)
axes[1,1].scatter(X4, lin_reg_y_total_area4, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,1].plot(X4, lin_reg_y_total_area4)
axes[1,1].plot(X4, m4*X4 + b4, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,1].set_xlabel('Set_point')
axes[1,1].set_ylabel('Delta_SOT')
axes[1,1].set_title('Linear Regression: Set point vs area (area > 200 mm^2)')
axes[1,1].grid()
axes[1,1].legend()

# Plot the results (visualization of Linear regression model vs actual data and the SPEC results for only test Datasets)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)


axes[0,0].scatter(X_test_area1, y_test_area1,color='blue', label='test data', s=100, alpha=0.5)
axes[0,0].scatter(X_test_area1, lin_reg_y_predicted_area1, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,0].plot(X_test_area1, lin_reg_y_predicted_area1)
axes[0,0].plot(X_test_area1, m1*X_test_area1 + b1, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,0].set_xlabel('Set_point')
axes[0,0].set_ylabel('Delta_SOT')
axes[0,0].set_title('Linear Regression: Set point vs area (area = 0-45 mm^2)')
axes[0,0].grid()
axes[0,0].legend()

axes[0,1].scatter(X_test_area2, y_test_area2,color='blue', label='test data', s=100, alpha=0.5)
axes[0,1].scatter(X_test_area2, lin_reg_y_predicted_area2, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,1].plot(X_test_area2, lin_reg_y_predicted_area2)
axes[0,1].plot(X_test_area2, m2*X_test_area2 + b2, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,1].set_xlabel('Set_point')
axes[0,1].set_ylabel('Delta_SOT')
axes[0,1].set_title('Linear Regression: Set point vs area (area = 45-100 mm^2)')
axes[0,1].grid()
axes[0,1].legend()

axes[1,0].scatter(X_test_area3, y_test_area3,color='blue', label='test data', s=100, alpha=0.5)
axes[1,0].scatter(X_test_area3, lin_reg_y_predicted_area3, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,0].plot(X_test_area3, lin_reg_y_predicted_area3)
axes[1,0].plot(X_test_area3, m3*X_test_area3 + b3, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,0].set_xlabel('Set_point')
axes[1,0].set_ylabel('Delta_SOT')
axes[1,0].set_title('Linear Regression: Set point vs area (area = 100-200 mm^2)')
axes[1,0].grid()
axes[1,0].legend()

axes[1,1].scatter(X_test_area4, y_test_area4,color='blue', label='test data', s=100, alpha=0.5)
axes[1,1].scatter(X_test_area4, lin_reg_y_predicted_area4, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,1].plot(X_test_area4, lin_reg_y_predicted_area4)
axes[1,1].plot(X_test_area4, m4*X_test_area4 + b4, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,1].set_xlabel('Set_point')
axes[1,1].set_ylabel('Delta_SOT')
axes[1,1].set_title('Linear Regression: Set point vs area (area > 200 mm^2)')
axes[1,1].grid()
axes[1,1].legend()

# Plot the results (visualization of Linear regression model vs actual data and the SPEC results for only train Datasets)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.5)


axes[0,0].scatter(X_train_area1, y_train_area1,color='blue', label='train data', s=100, alpha=0.5)
axes[0,0].scatter(X_train_area1, lin_reg_y_train_area1, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,0].plot(X_train_area1, lin_reg_y_train_area1)
axes[0,0].plot(X_train_area1, m1*X_train_area1 + b1, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,0].set_xlabel('Set_point')
axes[0,0].set_ylabel('Delta_SOT')
axes[0,0].set_title('Linear Regression: Set point vs area (area = 0-45 mm^2)')
axes[0,0].grid()
axes[0,0].legend()

axes[0,1].scatter(X_train_area2, y_train_area2,color='blue', label='train data', s=100, alpha=0.5)
axes[0,1].scatter(X_train_area2, lin_reg_y_train_area2, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[0,1].plot(X_train_area2, lin_reg_y_train_area2)
axes[0,1].plot(X_train_area2, m2*X_train_area2 + b2, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[0,1].set_xlabel('Set_point')
axes[0,1].set_ylabel('Delta_SOT')
axes[0,1].set_title('Linear Regression: Set point vs area (area = 45-100 mm^2)')
axes[0,1].grid()
axes[0,1].legend()

axes[1,0].scatter(X_train_area3, y_train_area3,color='blue', label='train data', s=100, alpha=0.5)
axes[1,0].scatter(X_train_area3, lin_reg_y_train_area3, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,0].plot(X_train_area3, lin_reg_y_train_area3)
axes[1,0].plot(X_train_area3, m3*X_train_area3 + b3, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,0].set_xlabel('Set_point')
axes[1,0].set_ylabel('Delta_SOT')
axes[1,0].set_title('Linear Regression: Set point vs area (area = 100-200 mm^2)')
axes[1,0].grid()
axes[1,0].legend()

axes[1,1].scatter(X_train_area4, y_train_area4,color='blue', label='train data', s=100, alpha=0.5)
axes[1,1].scatter(X_train_area4, lin_reg_y_train_area4, c='yellow', label='Regression values',linewidths=2, marker='^',edgecolor='red', s=200)
axes[1,1].plot(X_train_area4, lin_reg_y_train_area4)
axes[1,1].plot(X_train_area4, m3*X_train_area4 + b4, label='SPEC values')
#axes[0,0].set_ylim(-5.0, 8.0)
axes[1,1].set_xlabel('Set_point')
axes[1,1].set_ylabel('Delta_SOT')
axes[1,1].set_title('Linear Regression: Set point vs area (area > 200 mm^2)')
axes[1,1].grid()
axes[1,1].legend()


plt.show()