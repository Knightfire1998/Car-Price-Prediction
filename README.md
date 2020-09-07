# Car-price-prediction

#### Dataset for the model can be found at https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho
# Work Done:

On basis of features such as : 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual','Transmission_Manual', 'No_of_Year' the selling price of the vehicle is predicted

    1.Dataset was analyzed using graphs  heatmap, barplots and histograms for data insights.  
    2.No_of_year feature was derived from Year column of the orignal Dataset.  
    3.Preprocessing of categorical features was done using OneHotEncoding.  
    4.Feature importance was calculated using ExtraTreesRegressor and correlation matrix was used for feature selection.  
    5.HyperParameter tuning was done for selection of best parameters for RandomForestRegressor.  
    6.Model was fit to the RandomforestRegressor and tested on Test Dataset.  

# To Run the model:

Use requirements.txt for installing dependencies.  
Make a folder named as 'Data' in the working directory and put all the csv files in that folder  
Run the caprediction.py

# Additional woek (Deployment):
    
    1.Created a web-based application for car price prediction using FLASK
    2.Designed the web page using Html, Bootstrap.
    3.Can be deployed on any web servers such as Heroku, AWS, Azure.
    4.It predicts the selling price of the car based on the before mentioned attributes and displays it on the same page.
    4.This app is deployed on Heroku.
#### Link of the heroku app https://car-price-estimator-v1.herokuapp.com/
