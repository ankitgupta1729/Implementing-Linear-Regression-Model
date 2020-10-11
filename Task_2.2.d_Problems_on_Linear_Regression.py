#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd


# In[2]:


#reading .csv file and storing it in "dataset" dataframe
dataset=pd.read_csv("/home/ankit/Desktop/ex1data2.csv")


# In[3]:


print(dataset)


# In[4]:


print(dataset.shape) # size of the dataset


# In[5]:


data = dataset.to_numpy() # storing dataset as numpy array


# In[6]:


#To take separate values of both features in lists x and y
x1=[]
y1=[]
z1=[]
for i in range(len(data)):
    x1.append(data[i][0])
for i in range(len(data)):
    y1.append(data[i][1]) 
for i in range(len(data)):
    z1.append(data[i][2])  

# To normalize the data to put it on same scale
norm = np.linalg.norm(x1)
x=x1/norm
norm1 = np.linalg.norm(y1)
y=y1/norm1
norm2 = np.linalg.norm(z1)
z=z1/norm2



print(x)
print(y)
print(z) #actual value of z in Z=ax+by + c


# In[7]:


# Applying 10-fold cross-validation

#1st shuffle the data
import random
random.shuffle(x)
random.shuffle(y)
random.shuffle(z)
#splitting the shuffled data of 1st feature into 10 groups



fold1_x=[]
fold2_x=[]
fold3_x=[]
fold4_x=[]
fold5_x=[]
fold6_x=[]
fold7_x=[]
fold8_x=[]
fold9_x=[]
fold10_x=[]
for i in range(0,4):
    fold1_x.append(x[i])
for i in range(4,8):
    fold2_x.append(x[i])    
for i in range(8,12):
    fold3_x.append(x[i]) 
for i in range(12,17):
    fold4_x.append(x[i]) 
for i in range(17,22):
    fold5_x.append(x[i])   
for i in range(22,27):
    fold6_x.append(x[i]) 
for i in range(27,32):
    fold7_x.append(x[i])
for i in range(32,37):
    fold8_x.append(x[i])
for i in range(37,42):
    fold9_x.append(x[i])  
for i in range(42,47):
    fold10_x.append(x[i])
    
    
fold1_y=[]
fold2_y=[]
fold3_y=[]
fold4_y=[]
fold5_y=[]
fold6_y=[]
fold7_y=[]
fold8_y=[]
fold9_y=[]
fold10_y=[]
for i in range(0,4):
    fold1_y.append(y[i])
for i in range(4,8):
    fold2_y.append(y[i])    
for i in range(8,12):
    fold3_y.append(y[i]) 
for i in range(12,17):
    fold4_y.append(y[i]) 
for i in range(17,22):
    fold5_y.append(y[i])   
for i in range(22,27):
    fold6_y.append(y[i]) 
for i in range(27,32):
    fold7_y.append(y[i])
for i in range(32,37):
    fold8_y.append(y[i])
for i in range(37,42):
    fold9_y.append(y[i])  
for i in range(42,47):
    fold10_y.append(y[i])
    
    
fold1_z=[]
fold2_z=[]
fold3_z=[]
fold4_z=[]
fold5_z=[]
fold6_z=[]
fold7_z=[]
fold8_z=[]
fold9_z=[]
fold10_z=[]
for i in range(0,4):
    fold1_z.append(z[i])
for i in range(4,8):
    fold2_z.append(z[i])    
for i in range(8,12):
    fold3_z.append(z[i]) 
for i in range(12,17):
    fold4_z.append(z[i]) 
for i in range(17,22):
    fold5_z.append(z[i])   
for i in range(22,27):
    fold6_z.append(z[i]) 
for i in range(27,32):
    fold7_z.append(z[i])
for i in range(32,37):
    fold8_z.append(z[i])
for i in range(37,42):
    fold9_z.append(z[i])  
for i in range(42,47):
    fold10_z.append(z[i])    
    
for i in range(47):
    print(fold1_x,fold2_x,fold3_x,fold4_x,fold5_x,fold6_x,fold7_x,fold8_x,fold9_x,fold10_x)
for i in range(47):
    print(fold1_y,fold2_y,fold3_y,fold4_y,fold5_y,fold6_y,fold7_y,fold8_y,fold9_y,fold10_y)
for i in range(47):
    print(fold1_y,fold2_y,fold3_y,fold4_y,fold5_y,fold6_y,fold7_y,fold8_y,fold9_y,fold10_y)    


# In[8]:


len(fold10_x)
len(fold10_y)
len(fold10_z)


# In[9]:


#Now we define the gradient descent method to build the model 

def gradient_descent(x,y):

    a=0.45
    b=0.35
    c=0.33 # initializing unknown parameters which we have to find with some random values

    l = 0.001 # assigning learning rate as 0.0001 for good accuracy

    iterations = 1000 # initializaing number of iterations

    n= 47 # total number of datapoints

# Here, error function is sum of squared error function which is E = 1/2 sum(i=0 to n) (z_actual-z_pred)^2
# where z_pred is ax_i + by_i + c




    for p in range(iterations):
        sum1=0
        D_a=0
        D_b=0
        D_c=0
        for i in range(10):
             z_pred = a*x[i]+b*y[i]+c
             #print(z_pred)
             sum1 += (z[i]-z_pred)**2 # Calculating total sum of squared error(sse)
             #print(sum1)
             sse = sum1/2
             #print(sse)
             if sse < 0.02: # considering 0.02 is close to zero
                    break
        for i in range(10):
            z_pred = a*x[i]+b*y[i]+c
            #print(z_pred)
            D_a += (-1)*x[i]*(z[i] - z_pred)    #partial derivative of error function wrt a
            #print(D_a)
            D_b += (-1)*y[i]*(z[i] - z_pred)    #partial derivative of error function wrt b
            #print(D_b)
            D_c += (-1)*(z[i] - z_pred)         #partial derivative of error function wrt c
            #print(D_c)
        
            # Adjust the weights with the gradients to reach the optimal values where SSE is minimized
            
        a = a - l*D_a
        b = b - l*D_b
        c = c - l*D_c
        #print(a,b,c)
        # we will use these updated value for next iteration and we do this untill loss ~ 0
        #print("At iteration %d, The value of sse is: %2.5f  " %(p,sse))
        #print("final sum of squared error using gradient descent : ", sse)
        #print("Finalvalues of a,b,c are: ",a,b,c)        
        return a,b,c


# In[10]:


# Now, we take 9 folds data for training and one fold data for testing
d1=fold1_x + fold2_x + fold3_x + fold4_x + fold5_x + fold6_x + fold7_x + fold8_x + fold9_x
d2=fold1_y + fold2_y + fold3_y + fold4_y + fold5_y + fold6_y + fold7_y + fold8_y + fold9_y


# In[11]:


(a,b,c)=gradient_descent(d1,d2) # to get the hyperplane parameters for d1 and d2 data

print(a,b,c)


# In[12]:


# So, equation of hyperplane is z=0.4495745592685494*x + 0.34947106271956413*y +  0.32670808584178807
# Now, we test the model on 10th fold data and calculate the mean squared error
sum=0
for i in range(5):
    z_predict= 0.4495745592685494*fold10_x[i] + 0.34947106271956413*fold10_y[i] + 0.32670808584178807
    z_actual=fold10_z[i]
    sum +=(z_actual-z_predict)**2
error=sum/5    


# In[13]:


print(error) # it is mean-squared error when we take 1st 9 folds data as training and 10th fold data as test.
  


# In[14]:


#simialrly by taking 9 combinations of folding data for training and remaining 1 fold data for test 
#and calculate the mean squared error   

