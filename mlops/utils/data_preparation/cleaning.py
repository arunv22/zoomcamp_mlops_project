from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocessing(df):

    x = df.loc[:, ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']]
    y = df['MEDV']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Separate features and target for train and test sets
    x_train = train_df.loc[:, ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']]
    y_train = train_df['MEDV']
    x_test = test_df.loc[:, ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']]
    y_test = test_df['MEDV']
    # Apply log transformation to the target variable
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # Check skewness and apply log transformation if necessary
    for col in x_train.columns:
        if np.abs(x_train[col].skew()) > 0.3:
            x_train[col] = np.log1p(x_train[col])
            x_test[col] = np.log1p(x_test[col])

    # Fit the scaler on the training data and transform both train and test data
    min_max_scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(data=min_max_scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(data=min_max_scaler.transform(x_test), columns=x_test.columns)

    return x, x_train_scaled, x_test_scaled, y, y_train, y_test