# Covid-19 Project
## About the project 
This is a complete project about COVID-19 data visualization and prediction.
For data visualization we use clasic tools like pie-charts, heat-tables, graphs but also geographic heatmaps. 
For the prediction purposes, we focalise in cases and deaths caused from the pandemic and we use regression models
such as polynomial regression, support vector regression or ARIMA but also other forecasting methods such as 
Meta's Prophet. Machine/Deep learning models (neural networks) like LSTMs, GRUs, with Attention mechanisms are also deployed. 

The project is divided into two sub-parts: 1) The pre-processing, visualization and classical ways of prediction the pandemic and 2) The prediction and experimentation with neural networks. 
For more information you can read the Thesis short description [here](./Thesis_Description_English.pdf). Additionally, as the whole project was carried out in Googgle Colab, it is _highly recommended_ to run everything in Colab.

### Part 1: Pre-processing & Data Visualization & Classical Prediction.

__STEP 1__: Clone the repository by using: 
``` bash
git clone https://github.com/arpajj/Covid-19-Project.git
```
__STEP 2__: Upload and store everything (all files) in your Google Drive under the same directory.

__STEP 3__: Start running the the notebook [firt_part.ipynb](./first_part.ipynb) for exploring & visualizing the Covid-19 data and testing the initial methods of prediction.


### Part 2: Experimentation & Prediction with Neural Networks.

__STEP 1__: (If you haven't from Part 1) Store the folder [data-second-phase](./data-second-phase) and the notebook [second_part.ipynb](./second_part.ipynb) under the same directory in Google Drive.

__STEP 2__: Start experimenting with the notebook [second_part.ipynb](./second_part.ipynb), for running experiments with different data and models. Examine the data (dimensions, windows, slidings etc) and models (different types, (hyper)-parameters, adding attention etc). Refer to the main notebook of the Thesis 
([final_version.ipynb](./final_version.ipynb) - executed back in 2021), for understanding and see more about the technicalities of each approach.



