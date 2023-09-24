from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def get_model():

    lr_model = LinearRegression()
    return lr_model

# Create polynomial features
poly = PolynomialFeatures(degree=2)

# Scale features
scaler = StandardScaler()

# Initialize Ridge Regression model with regularization strength alpha
ridge_model = Ridge(alpha=1.0)

# Create a pipeline with polynomial feature creation, scaling, and Ridge Regression
model = make_pipeline(poly, scaler, ridge_model)

# # Train the model with training data
# model.fit(X_train, y_train)

# # Make predictions on validation data
# y_val_pred = model.predict(X_val)

# # Calculate Mean Absolute Error on validation data
# mae_val = mean_absolute_error(y_val, y_val_pred)

if __name__ == "__main__":
    lr = get_model()
    print(lr)