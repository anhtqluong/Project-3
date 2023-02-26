# Overview

This project analyses the 2020 used car data to identify what features have a strong relationship with resale price of used car. The analysis also looks to dertermine if the model can be used to accurately predict the resale price. 

# Data Understanding and Preparation


```python
# Import Standard Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.formula.api import ols


sales = pd.read_csv('data/car data.csv')
pd.options.display.float_format = '{:,.0f}'.format
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ritz</td>
      <td>2014</td>
      <td>3</td>
      <td>6</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sx4</td>
      <td>2013</td>
      <td>5</td>
      <td>10</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ciaz</td>
      <td>2017</td>
      <td>7</td>
      <td>10</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wagon r</td>
      <td>2011</td>
      <td>3</td>
      <td>4</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>swift</td>
      <td>2014</td>
      <td>5</td>
      <td>7</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 301 entries, 0 to 300
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Car_Name       301 non-null    object 
     1   Year           301 non-null    int64  
     2   Selling_Price  301 non-null    float64
     3   Present_Price  301 non-null    float64
     4   Kms_Driven     301 non-null    int64  
     5   Fuel_Type      301 non-null    object 
     6   Seller_Type    301 non-null    object 
     7   Transmission   301 non-null    object 
     8   Owner          301 non-null    int64  
    dtypes: float64(2), int64(3), object(4)
    memory usage: 21.3+ KB
    


```python
#check for any NAN value
sales.isna().sum()
```




    Car_Name         0
    Year             0
    Selling_Price    0
    Present_Price    0
    Kms_Driven       0
    Fuel_Type        0
    Seller_Type      0
    Transmission     0
    Owner            0
    dtype: int64



Based on the above, Waterfront and Yr Renovated require data cleasning 


```python
#add "age" column 
sales['Age']=[2020] - sales['Year']
sales.drop(['Year','Car_Name'],axis=1,inplace = True)
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>6</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>10</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 301 entries, 0 to 300
    Data columns (total 8 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Selling_Price  301 non-null    float64
     1   Present_Price  301 non-null    float64
     2   Kms_Driven     301 non-null    int64  
     3   Fuel_Type      301 non-null    object 
     4   Seller_Type    301 non-null    object 
     5   Transmission   301 non-null    object 
     6   Owner          301 non-null    int64  
     7   Age            301 non-null    int64  
    dtypes: float64(2), int64(3), object(3)
    memory usage: 18.9+ KB
    

# Outlier Detection and Removal


```python
#outlier detection
import seaborn as sns
sns.set_style('darkgrid')
colors = ['#0055ff', '#ff7000', '#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))

OrderedCols = np.concatenate([sales.select_dtypes(exclude='object').columns.values, 
                              sales.select_dtypes(include='object').columns.values])

fig, ax = plt.subplots(2, 4, figsize=(15,7),dpi=100)

for i,col in enumerate(OrderedCols):
    x = i//4
    y = i%4
    if i<5:
        sns.boxplot(data=sales, y=col, ax=ax[x,y])
        ax[x,y].yaxis.label.set_size(15)
    else:
        sns.boxplot(data=sales, x=col, y='Selling_Price', ax=ax[x,y])
        ax[x,y].xaxis.label.set_size(15)
        ax[x,y].yaxis.label.set_size(15)

plt.tight_layout()    
plt.show()
```


    
![png](output_10_0.png)
    



```python
outliers_indexes = []
target = 'Selling_Price'

for col in sales.select_dtypes(include='object').columns:
    for cat in sales[col].unique():
        df1 = sales[sales[col] == cat]
        q1 = df1[target].quantile(0.25)
        q3 = df1[target].quantile(0.75)
        iqr = q3-q1
        maximum = q3 + (1.5 * iqr)
        minimum = q1 - (1.5 * iqr)
        outlier_samples = df1[(df1[target] < minimum) | (df1[target] > maximum)]
        outliers_indexes.extend(outlier_samples.index.tolist())
        
        
for col in sales.select_dtypes(exclude='object').columns:
    q1 = sales[col].quantile(0.25)
    q3 = sales[col].quantile(0.75)
    iqr = q3-q1
    maximum = q3 + (1.5 * iqr)
    minimum = q1 - (1.5 * iqr)
    outlier_samples = sales[(sales[col] < minimum) | (sales[col] > maximum)]
    outliers_indexes.extend(outlier_samples.index.tolist())
    
outliers_indexes = list(set(outliers_indexes))
print('{} outliers were identified, whose indices are:\n\n{}'.format(len(outliers_indexes), outliers_indexes))
```

    38 outliers were identified, whose indices are:
    
    [27, 37, 39, 50, 51, 52, 53, 54, 179, 184, 58, 59, 189, 62, 63, 64, 191, 66, 192, 196, 69, 193, 198, 201, 77, 205, 79, 80, 82, 84, 85, 86, 92, 93, 96, 97, 106, 241]
    


```python
# Outliers Labeling
df1 = sales.copy()
df1['label'] = 'Normal'
df1.loc[outliers_indexes,'label'] = 'Outlier'

# Removing Outliers
removing_indexes = []
removing_indexes.extend(df1[df1[target]>33].index)
removing_indexes.extend(df1[df1['Kms_Driven']>400000].index)
df1.loc[removing_indexes,'label'] = 'Removing'

# Plot
target = 'Selling_Price'
features = sales.columns.drop(target)
colors = ['#0055ff','#ff7000','#23bf00']
CustomPalette = sns.set_palette(sns.color_palette(colors))
fig, ax = plt.subplots(nrows=3 ,ncols=3, figsize=(15,12), dpi=200)

for i in range(len(features)):
    x=i//3
    y=i%3
    sns.scatterplot(data=df1, x=features[i], y=target, hue='label', ax=ax[x,y])
    ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 15)
    ax[x,y].set_xlabel(features[i], size = 12)
    ax[x,y].set_ylabel(target, size = 12)
    ax[x,y].grid()

ax[2, 1].axis('off')
ax[2, 2].axis('off')
plt.tight_layout()
plt.show()
```


    
![png](output_12_0.png)
    



```python
df1 = sales.copy()
df1.drop(removing_indexes, inplace=True)
df1.reset_index(drop=True, inplace=True)
```

# Explore Variables


```python
#Check categorical variables

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,10), sharey=True)

categoricals = ['Fuel_Type', 'Seller_Type', 'Transmission']

for col, ax in zip(categoricals, axes.flatten()):
    (df1.groupby(col)               # group values together by column of interest
         .mean()['Selling_Price']        # take the mean of the saleprice for each group
         .sort_values()              # sort the groups in ascending order
         .plot
         .bar(ax=ax))                # create a bar graph on the ax
    
    ax.set_title(col)                # Make the title the name of the column
    
fig.tight_layout()
```


    
![png](output_15_0.png)
    



```python
# check distribution of the continuous variables 
continuous = ['Age', 'Selling_Price', 'Kms_Driven','Present_Price']
df1_cont = df1[continuous]
pd.plotting.scatter_matrix(df1_cont, figsize=(10,12));
```


    
![png](output_16_0.png)
    


# Build Base Model


```python
#create dummies: 
CatCols = ['Fuel_Type', 'Seller_Type', 'Transmission','Owner']
df1 = pd.get_dummies(df1, columns=CatCols, drop_first=True)
df1.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Age</th>
      <th>Fuel_Type_Diesel</th>
      <th>Fuel_Type_Petrol</th>
      <th>Seller_Type_Individual</th>
      <th>Transmission_Manual</th>
      <th>Owner_1</th>
      <th>Owner_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>6</td>
      <td>27000</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>43000</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>10</td>
      <td>6900</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>5200</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>42450</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df1.drop('Selling_Price', axis=1)
y = df1['Selling_Price']
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>Selling_Price</td>  <th>  R-squared:         </th> <td>   0.894</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.890</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   270.1</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Feb 2023</td> <th>  Prob (F-statistic):</th> <td>4.51e-135</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:45:28</td>     <th>  Log-Likelihood:    </th> <td> -556.34</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   299</td>      <th>  AIC:               </th> <td>   1133.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   289</td>      <th>  BIC:               </th> <td>   1170.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>    3.1561</td> <td>    1.206</td> <td>    2.616</td> <td> 0.009</td> <td>    0.781</td> <td>    5.531</td>
</tr>
<tr>
  <th>Present_Price</th>          <td>    0.5584</td> <td>    0.021</td> <td>   26.608</td> <td> 0.000</td> <td>    0.517</td> <td>    0.600</td>
</tr>
<tr>
  <th>Kms_Driven</th>             <td>-2.348e-05</td> <td> 4.76e-06</td> <td>   -4.936</td> <td> 0.000</td> <td>-3.28e-05</td> <td>-1.41e-05</td>
</tr>
<tr>
  <th>Age</th>                    <td>   -0.2981</td> <td>    0.044</td> <td>   -6.848</td> <td> 0.000</td> <td>   -0.384</td> <td>   -0.212</td>
</tr>
<tr>
  <th>Fuel_Type_Diesel</th>       <td>    1.9769</td> <td>    1.149</td> <td>    1.721</td> <td> 0.086</td> <td>   -0.284</td> <td>    4.238</td>
</tr>
<tr>
  <th>Fuel_Type_Petrol</th>       <td>    0.3510</td> <td>    1.128</td> <td>    0.311</td> <td> 0.756</td> <td>   -1.870</td> <td>    2.572</td>
</tr>
<tr>
  <th>Seller_Type_Individual</th> <td>   -0.4630</td> <td>    0.252</td> <td>   -1.834</td> <td> 0.068</td> <td>   -0.960</td> <td>    0.034</td>
</tr>
<tr>
  <th>Transmission_Manual</th>    <td>   -0.5476</td> <td>    0.321</td> <td>   -1.708</td> <td> 0.089</td> <td>   -1.179</td> <td>    0.084</td>
</tr>
<tr>
  <th>Owner_1</th>                <td>    0.3925</td> <td>    0.518</td> <td>    0.758</td> <td> 0.449</td> <td>   -0.626</td> <td>    1.411</td>
</tr>
<tr>
  <th>Owner_3</th>                <td>   -6.2870</td> <td>    1.678</td> <td>   -3.747</td> <td> 0.000</td> <td>   -9.589</td> <td>   -2.985</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>54.596</td> <th>  Durbin-Watson:     </th> <td>   1.857</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 393.860</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.473</td> <th>  Prob(JB):          </th> <td>2.98e-86</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.543</td> <th>  Cond. No.          </th> <td>9.68e+05</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 9.68e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
#Check Residuals 
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_20_0.png)
    


We noted that the continuous variables were right skewed. 


```python
# log features
log_names = [f'{column}_log' for column in df1_cont.columns]

df2_log = np.log(df1_cont)
df2_log.columns = log_names

df2_log.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_log</th>
      <th>Selling_Price_log</th>
      <th>Kms_Driven_log</th>
      <th>Present_Price_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.plotting.scatter_matrix(df2_log, figsize=(10,12));
```


    
![png](output_23_0.png)
    



```python
df3 = pd.concat([df2_log,df1], axis=1)
df3.drop(['Age', 'Selling_Price', 'Kms_Driven','Present_Price'],axis=1,inplace = True)
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_log</th>
      <th>Selling_Price_log</th>
      <th>Kms_Driven_log</th>
      <th>Present_Price_log</th>
      <th>Fuel_Type_Diesel</th>
      <th>Fuel_Type_Petrol</th>
      <th>Seller_Type_Individual</th>
      <th>Transmission_Manual</th>
      <th>Owner_1</th>
      <th>Owner_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rerun the model 
X = df3.drop('Selling_Price_log', axis=1)
y = df3['Selling_Price_log']
```


```python
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Selling_Price_log</td> <th>  R-squared:         </th> <td>   0.976</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.975</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   1316.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Feb 2023</td>  <th>  Prob (F-statistic):</th> <td>8.66e-229</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:45:29</td>      <th>  Log-Likelihood:    </th> <td>  66.566</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   299</td>       <th>  AIC:               </th> <td>  -113.1</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   289</td>       <th>  BIC:               </th> <td>  -76.13</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>       <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>     <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>    1.3081</td> <td>    0.201</td> <td>    6.496</td> <td> 0.000</td> <td>    0.912</td> <td>    1.704</td>
</tr>
<tr>
  <th>Age_log</th>                <td>   -0.6796</td> <td>    0.036</td> <td>  -18.934</td> <td> 0.000</td> <td>   -0.750</td> <td>   -0.609</td>
</tr>
<tr>
  <th>Kms_Driven_log</th>         <td>   -0.0542</td> <td>    0.016</td> <td>   -3.311</td> <td> 0.001</td> <td>   -0.086</td> <td>   -0.022</td>
</tr>
<tr>
  <th>Present_Price_log</th>      <td>    0.9071</td> <td>    0.023</td> <td>   40.288</td> <td> 0.000</td> <td>    0.863</td> <td>    0.951</td>
</tr>
<tr>
  <th>Fuel_Type_Diesel</th>       <td>    0.2464</td> <td>    0.143</td> <td>    1.727</td> <td> 0.085</td> <td>   -0.034</td> <td>    0.527</td>
</tr>
<tr>
  <th>Fuel_Type_Petrol</th>       <td>    0.0799</td> <td>    0.141</td> <td>    0.568</td> <td> 0.570</td> <td>   -0.197</td> <td>    0.357</td>
</tr>
<tr>
  <th>Seller_Type_Individual</th> <td>   -0.2261</td> <td>    0.053</td> <td>   -4.278</td> <td> 0.000</td> <td>   -0.330</td> <td>   -0.122</td>
</tr>
<tr>
  <th>Transmission_Manual</th>    <td>    0.0270</td> <td>    0.036</td> <td>    0.750</td> <td> 0.454</td> <td>   -0.044</td> <td>    0.098</td>
</tr>
<tr>
  <th>Owner_1</th>                <td>   -0.1788</td> <td>    0.064</td> <td>   -2.778</td> <td> 0.006</td> <td>   -0.305</td> <td>   -0.052</td>
</tr>
<tr>
  <th>Owner_3</th>                <td>   -0.6812</td> <td>    0.212</td> <td>   -3.218</td> <td> 0.001</td> <td>   -1.098</td> <td>   -0.265</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>19.284</td> <th>  Durbin-Watson:     </th> <td>   1.754</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  31.283</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.414</td> <th>  Prob(JB):          </th> <td>1.61e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.351</td> <th>  Cond. No.          </th> <td>    241.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#Check the distribution of residuals 

import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_27_0.png)
    



```python
#drop ages and check if it has improved the model R square 
df3= df3.drop(['Fuel_Type_Diesel','Fuel_Type_Petrol','Transmission_Manual'], axis=1)
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_log</th>
      <th>Selling_Price_log</th>
      <th>Kms_Driven_log</th>
      <th>Present_Price_log</th>
      <th>Seller_Type_Individual</th>
      <th>Owner_1</th>
      <th>Owner_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rerun the model 
X = df3.drop('Selling_Price_log', axis=1)
y = df3['Selling_Price_log']
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>Selling_Price_log</td> <th>  R-squared:         </th> <td>   0.974</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>        <th>  Adj. R-squared:    </th> <td>   0.973</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>   <th>  F-statistic:       </th> <td>   1824.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Feb 2023</td>  <th>  Prob (F-statistic):</th> <td>3.73e-228</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:45:29</td>      <th>  Log-Likelihood:    </th> <td>  53.515</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   299</td>       <th>  AIC:               </th> <td>  -93.03</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   292</td>       <th>  BIC:               </th> <td>  -67.13</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>       <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>     <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                  <td>    1.2679</td> <td>    0.141</td> <td>    8.990</td> <td> 0.000</td> <td>    0.990</td> <td>    1.545</td>
</tr>
<tr>
  <th>Age_log</th>                <td>   -0.7096</td> <td>    0.037</td> <td>  -19.286</td> <td> 0.000</td> <td>   -0.782</td> <td>   -0.637</td>
</tr>
<tr>
  <th>Kms_Driven_log</th>         <td>   -0.0369</td> <td>    0.017</td> <td>   -2.220</td> <td> 0.027</td> <td>   -0.070</td> <td>   -0.004</td>
</tr>
<tr>
  <th>Present_Price_log</th>      <td>    0.9365</td> <td>    0.021</td> <td>   43.620</td> <td> 0.000</td> <td>    0.894</td> <td>    0.979</td>
</tr>
<tr>
  <th>Seller_Type_Individual</th> <td>   -0.1941</td> <td>    0.053</td> <td>   -3.649</td> <td> 0.000</td> <td>   -0.299</td> <td>   -0.089</td>
</tr>
<tr>
  <th>Owner_1</th>                <td>   -0.1697</td> <td>    0.067</td> <td>   -2.540</td> <td> 0.012</td> <td>   -0.301</td> <td>   -0.038</td>
</tr>
<tr>
  <th>Owner_3</th>                <td>   -0.8130</td> <td>    0.218</td> <td>   -3.730</td> <td> 0.000</td> <td>   -1.242</td> <td>   -0.384</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21.804</td> <th>  Durbin-Watson:     </th> <td>   1.812</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  32.130</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.500</td> <th>  Prob(JB):          </th> <td>1.05e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.257</td> <th>  Cond. No.          </th> <td>    195.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_30_0.png)
    


QQ plot shows that the normality assumption of the residuals seems fulfilled.


```python
#check for Homoscedasticity

fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age_log", fig=fig)
plt.show()
```


    
![png](output_32_0.png)
    



```python
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Kms_Driven_log", fig=fig)
plt.show()
```


    
![png](output_33_0.png)
    



```python
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Present_Price_log", fig=fig)
plt.show()
```


    
![png](output_34_0.png)
    


Based on the above, the assumption of Homoscedasticity appears to be met.

Train-Test Split:
Perform a train-test split with a test set of 20% and a random state of 4.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```


```python
#Fit a linear regression model on the training set

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

```




    LinearRegression()




```python
#Generate Predictions on Training and Test Sets
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```


```python
#Calculate the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error: ', test_mse)
```

    Train Mean Squared Error: 0.04389259098323844
    Test Mean Squared Error:  0.029247365996622747
    

The difference between Test MSE and Train MSE is a little large(appx 33%)


```python
# Import relevant modules and functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Transform with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Scale the test set
X_test_scaled = scaler.transform(X_test)

poly= PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.fit_transform(X_test_scaled)

# Check the shape
np.shape(X_train_poly)
```




    (239, 27)




```python
#fit regression into the model: 
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
```




    LinearRegression()




```python
# Training set predictions
poly_train_predictions = polyreg.predict(X_train_poly)

# Test set predictions 
poly_test_predictions = polyreg.predict(X_test_poly)
```


```python
#Calculate the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
poly_train_mse = mean_squared_error(y_train, poly_train_predictions)
poly_test_mse = mean_squared_error(y_test, poly_test_predictions)
print('Train Mean Squared Error:', poly_train_mse)
print('Test Mean Squared Error: ', poly_test_mse)
```

    Train Mean Squared Error: 0.025801024134270403
    Test Mean Squared Error:  0.03341965250967328
    

The variance between test and train mean squared error has imprved (now ~ 29%). The overvall error is very low. 


```python
#Plot predictions for the training set against the actual data:
plt.figure(figsize=(8, 5))
plt.scatter(y_train, poly_train_predictions, label='Model')
plt.plot(y_train, y_train, label='Actual data')
plt.title('Model vs data for training set')
plt.legend();
```


    
![png](output_47_0.png)
    



```python
#Plot predictions for the test set against the actual data:
plt.figure(figsize=(8, 5))
plt.scatter(y_test, poly_test_predictions, label='Model')
plt.plot(y_test, y_test, label='Actual data')
plt.title('Model vs data for test set')
plt.legend();
```


    
![png](output_48_0.png)
    


Both train and test model are fitted well. 

# Interpretation 

The below features have the most impact when it comes to resale pricing for used cars: 

Age, Km driven, Seller Type and number of owners have a negative impacts on the price of used car, meaning that any increase in one of these features, will result in a decrease of the resale price of an used car. 

On the other hand, present price of the used car has a positive impact of the resale price, meaning that the higher the current price is, the higher the resale price will be. 


Model: 

The model R square is 0.973 - this mean that 97.3% of the resale price of an used car can be explained by the dependent variables in the model.

Mean square error (MSE) of the train test plit was improved after adding polynomial features into the data. Eventhough the difference between MSE train and test is 29%, both MSEs are quite low and close to 0. This suggests that the model has low level of error. 

In conclusion, the model is fitted quite well and can be used to predict the price highly accurately. 
