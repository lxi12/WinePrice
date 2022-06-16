"""Process the data, Merge excel files into one master spreadsheet;
   data visulisation check linearity;
   build the model to predict the wine price;
   different approaches to improve the model;
   Build the GUI allow users to click wine variety and enter wine score;
   Output suggested wine price based on the model.
   
@ author lxi12   
 """
#please import below packages, there might other packages need to installed as well
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


""" Data Processing"""
cwd = os.path.abspath('') #current working directory
files = os.listdir(cwd)   #files inside the cwd

df = pd.DataFrame()       #create empty dataframe
for file in files:        #loop through the files in cwd
    if file.endswith('.xlsx'):
        df = df.append(pd.read_excel(file), ignore_index=True)
 
df.to_excel("train_wine.xlsx", index = False)
#come out not perfect, some empty rows, two same cols, probably some ASCII characters invisble.
#manually process the data and change encoding to "utf-8", saved as "total.csv"

filename = "total.csv"
df = pd.read_csv(filename,sep=",") #separate to columns by \t
df.head()
df.tail(n=2)
type(df)                          #object, pandas.dataframe, a dictionary
print(df.columns)                 #check column names

#reassign column headers
traindata = df.set_axis(["name", "grape", "popularity", "score", 
                          "price"], axis=1, inplace=False)
print(traindata.info())


"""Data Visulisation"""
#check for Linearity: price ~ score
scores = traindata["score"].values[:]
prices = traindata["price"].values[:]
a, b = np.polyfit(scores, prices, deg=1)
y_fitted = a * scores + b

#plot the data and model 
axes = plt.axes()
axes.plot(scores, prices, "bo")
axes.plot(scores, y_fitted, "r--")
axes.grid()
axes.set_xlabel("score")
axes.set_ylabel("price")
axes.set_title("Data Visiulisation: Price ~ Score")
plt.show()
#from plot, we can clear see there is a  slightly linear relation between score and price
#but model price go below 0, which need to fix 


#check for Linearity: price ~ popularity
populs = traindata["popularity"].values[:]
a, b = np.polyfit(populs, prices, deg=1)
y_fitted = a * populs + b

#plot the data and model 
axes = plt.axes()
axes.plot(populs, prices, "bo")
axes.plot(populs, y_fitted, "r--")
axes.grid()
axes.set_xlabel("popularity")
axes.set_ylabel("price")
axes.set_title("Data Visiulisation: Price ~ Popularity")
plt.show()
#from plot, we can see there is no significant linearity between price and popularity


#Boxplot price ~ grape
print(traindata["grape"].unique())
traindata.boxplot("price", "grape", rot=20, figsize=(5,6)) 
#the boxplot is different among different grape varieties, we could say grape variety contributes to price


"""Machine Learning, Building the model"""
"""perform multiple Linear Regression"""
#drop the wine name, and response variable price
xs= traindata.drop(["name", "price"], axis=1) 
ys = traindata["price"] 

#convert categorical variable to dummy variable
grapes= pd.get_dummies(xs, drop_first=True)
xs = xs.drop("grape", axis = 1)        #drop the categorical column
xs = pd.concat([xs, grapes], axis=1)   #concatation of independent variable and new categorical variable 
print(xs)                              #show 12 features

#create an object of LinearRegression class
MLR = linear_model.LinearRegression()
MLR.fit(xs, ys)
print("Intercept: \n", MLR.intercept_)
print("Coefficients: \n", MLR.coef_)


"""Build Ordinary Least Squares Regression"""
xs = sm.add_constant(xs)               #adding a constant to allow R^2 centering
model = sm.OLS(ys, xs).fit() 
print_model = model.summary()
print(print_model)
#comment on the output of this model:
#the intercept and coefficents results are the same with above MLR model
#with the R^2 value, it suggest 52.3% of the data can be explained by this model
#the model suggests strong multicollinearity problems, there is a correlation between explanatory variables, 
#it might because "score" and "popularity" show twice, not really know why and where I went wrong
#using different algorigthms to drop features, try to improve the model


"""calculating VIF for each feature"""

vif_data = pd.DataFrame()
vif_data["feature"] = xs.columns
vif_data["VIF"] = [variance_inflation_factor(xs.values, i) 
                   for i in range(len(xs.columns))]
print(vif_data)
#the vif values of popularity & score are "inf", this shows a perfect correlation
#between them. Decide to drop popularity, from linearity check, its influence is less

xs_p = xs.drop("popularity", axis=1)
model_p = sm.OLS(ys, xs_p).fit() 
print_model_p = model_p.summary()
print(print_model_p) 
#he model explains 52.1% of the data, a little bit lower than previoud model, and it still suggests multicollinearity problems 


"""Try another approach, using PCA to select the important features"""

pca = PCA(n_components=7)        #keep about half features
xs_t = pca.fit_transform(xs)     #apply the dimensionality reduction on xs
xs_7 = pd.DataFrame(data= xs_t, columns=["pca1", "pca2", "pca3", "pca4",
                                            "pca5", "pca6", "pca7"])
 
xs_7 = sm.add_constant(xs_7)  
model_7 = sm.OLS(ys, xs_7).fit() 
print_model_7 = model_7.summary()
print(print_model_7)
#The multicollinearity problem still exists
#and it suggests only about 45.5% of the data can be explained by this model, a bit less than above 


# To further improve the model
"""using Stepwise, drop p values > 0.05 feature"""

xs_6 = xs_7.drop("pca4", axis=1) #drop pca4
xs_6 = sm.add_constant(xs_6)  
model_6 = sm.OLS(ys, xs_6).fit() 
print_model_6 = model_6.summary()
print(print_model_6)

#it suggests only 45.5% of the data can be explained by this model,and multicollinearity promblem still exists
#And P values all <0.5 at this stage, we couldn't drop any feature at this stage.

"""Conclusion of model building process:
   After applying MLR, VIF, PCA. The model still indicates might exists strong multicollinearity problem.
   I would asuming because of feature "popularity" and "score" show up twice, also might exist other errors.
   
   Although in real-world practice, R^2 would be expected a bit higher. 
   In reality, wine price would be influenced by many other features, like wine country, wine brand, wine location, winemakers, year etc. 
   As expanatory variables are not small in this dataset, so the model result is not significant. 
  
   Conclusion: 
       despite this issue, I would suggest model_p after VIR algorithm has the best approach on this practice.
 
"""
mymodel=model_p

""" Create GUI using tkinter in Python"""

#the main window
root = tk.Tk()
canvas1 = tk.Canvas(root, width = 500, height = 350)
canvas1.pack()


label1=tk.Label(root,text="Hello, Wine Lovers",font=('Helvetica', 12, 'bold'),justify="center")
canvas1.create_window(250, 40, window=label1)

label2=tk.Label(root,text="Welcome to Wine Price Suggestion Site",font=('Helvetica', 10, 'bold'),justify="center") 
canvas1.create_window(250, 65, window=label2)


""" create global variables, tkvariables to hold value 0 or 1"""
global brut, CS, chard, merlot, PG, PN, riesl, SB, syrah
brut = tk.IntVar()
brut.set(0)

CS = tk.IntVar()
CS.set(0)

chard = tk.IntVar()
chard.set(0)

merlot = tk.IntVar()
merlot.set(0)

PG= tk.IntVar()
PG.set(0)

PN = tk.IntVar()
PN.set(0)

riesl = tk.IntVar()
riesl.set(0)

SB = tk.IntVar()
SB.set(0)

syrah = tk.IntVar()
syrah.set(0)

 
"""define functions when button is clicked, the appropriate variable change to 1"""

def count_brut():
    """output brut to 1 whenever the button "Brut" is pressed""" 
    brut.set(1)
    
def count_SB():
    """output SB to 1 whenever the button "Sauvignon Blanc" is pressed"""
    SB.set(1)
 
def count_chard():
    """output chard to 1 whenever the button "Chardonnay" is pressed"""
    chard.set(1)
 
def count_PG():
    """output PG to 1 whenever the button "Pinot Gris" is pressed"""
    PG.set(1)
 
def count_riesl():
    """output riesl to 1 whenever the button "Riesling" is pressed"""
    riesl.set(1)
  
def count_PN():
    """output PN to 1 whenever the button "Pinot Noir" is pressed"""
    PN.set(1)
 
def count_syrah():
    """output syrah to 1 whenever the button "Syrah" is pressed"""
    syrah.set(1)
 
def count_CS():
    """output CS to 1 whenever the button "Cabernet Sauvignon" is pressed"""
    CS.set(1)
    
def count_merlot():
    """output merlot to 1 whenever the button "Merlot" is pressed"""
    merlot.set(1)

    
"""create wine varieties buttons, and call above function if been clicked"""

button1 = tk.Button(root,text="Sauvignon Blanc",command=count_SB,padx=5,bg="white")
canvas1.create_window(100, 100, window=button1)

button2 = tk.Button(root,text="Chardonnay",command=count_chard,padx=5,bg="white")
canvas1.create_window(200, 100, window=button2)

button3 = tk.Button(root,text="Pinot Gris",command=count_PG,padx=5,bg="white")
canvas1.create_window(280, 100, window=button3)

button4 = tk.Button(root,text="Riesling",command=count_riesl,padx=5,bg="white")
canvas1.create_window(350, 100, window=button4)

button5 = tk.Button(root,text="Brut",command=count_brut,padx=5,bg="white")
canvas1.create_window(410, 100, window=button5) 
 
button6 = tk.Button(root,text="Pinot Noir",command=count_PN,padx=5,bg="red")
canvas1.create_window(70, 130, window=button6)

button7 = tk.Button(root,text="Syrah",command=count_syrah,padx=5,bg="red")
canvas1.create_window(135, 130, window=button7)

button8 = tk.Button(root,text="Bordeaux Blend",padx=5,bg="red")
canvas1.create_window(215, 130, window=button8)

button9 = tk.Button(root,text="Cabernet Sauvignon",command=count_CS,padx=5,bg="red")
canvas1.create_window(335, 130, window=button9)

button10 = tk.Button(root,text="Merlot",command=count_merlot,padx=5,bg="red")
canvas1.create_window(430, 130, window=button10) 
 

"""create score label and input box"""

label3 = tk.Label(root, text="Wine Score: ",font=('Helvetica', 10))
canvas1.create_window(150, 170, window=label3)
 
entry1 = tk.Entry (root)
canvas1.create_window(260, 170, window=entry1)
 

"""define function to read user's enter score value and respond to button click action,
   input these values into my model, output suggested wine price"""
 
def values():
    """the magic happens"""
    global score_rate
    score_rate = int(entry1.get())  
    
    df=pd.DataFrame([[1,score_rate,score_rate,brut.get(),CS.get(),chard.get(),merlot.get(),PG.get(),PN.get(),riesl.get(),SB.get(),syrah.get()]]) 
    predict_price = int(mymodel.predict(df))
    predict_price = max(predict_price, 12)    #as the model is not significant, to avoid output negative values, output at leaset data minimial price $12
    
    result = (f"Wine Price:  ${predict_price}")
    result_label = tk.Label(root, text = result, font=('Helvetica', 12, 'bold'),bg='orange')
    canvas1.create_window(260, 250, window=result_label) 
    

button11 = tk.Button(root, text = "Suggested Wine Price", command=values,padx=6,bg="orange")    
canvas1.create_window(260, 210, window=button11)


root.mainloop()

 
