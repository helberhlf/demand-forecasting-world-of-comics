#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing classes, to calculate the evaluation metrics of the predictive models.
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# Creating a function to evaluate the model for regression metrics
def regression(model,x_test,y_test):

    # Evaluating the model
    y_pred  = model.predict(x_test)
    print("Metrics in Test data:\n ")
    print('='*30)
    # Calculate the error
    print('MAE:', mean_absolute_error(y_test,y_pred))
    print('MAPE:',mean_absolute_percentage_error(y_test,y_pred))
    print('MSE:', mean_squared_error(y_test,y_pred))
    print('RMSE:',mean_squared_error(y_test,y_pred, squared = False))
    print('R2:',  r2_score(y_test, y_pred))


