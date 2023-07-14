import myspkmeanssp as spkmeans
import numpy as np
import pandas as pd
import sys

def quicksort(U, l, r):
   if(l<r):
      pivot,i,j = l,l,r
      while(i<j):
         while(U[i][0]<=U[pivot][0] and i<r):
            i+=1
         while(U[j][0]>U[pivot][0]):
            j-=1
         if(i<j):
            temp=U[i]
            U[i]=U[j]
            U[j]=temp 
      temp=U[pivot]
      U[pivot]=U[j]
      U[j]=temp
      quicksort(U,l,j-1)
      quicksort(U,j+1,r) 
       
def kmeans_pp(df, N, d, k):
    np.random.seed(0)
    index = np.random.choice(N) # generate rand index_0 . µ0 = df[index_0]  
    C = np.empty((0,d), float)  #centroids[i] = µi, where µi = df[miu_index], ∀i 0 ≤ i < k
    C = np.append(C, np.array([df[index]]), axis=0) # adding the rellevant vector to centroids
    C_indices = [index] # C_indices[0]= index_0
    
    i=1
    while(i<k):
        D = np.zeros(N) # D[l] = min{(xl − µj)^2}, ∀j 1≤j≤i
        P = np.zeros(N) # P[l] = D[l]/sum(D)
        for l in range(N): 
            D[l]= np.min([np.linalg.norm(df[l]-C[j])**2 for j in range(i)])
        for l in range(N): 
            P[l] = D[l]/np.sum(D)
        index = np.random.choice(N, p=P) # generate rand index_i according to P
        C_indices.append(index) # C_indices[i]= index_i
        C = np.append(C, np.array([df[index]]), axis=0) # C[i]=µi=df[index_i], ∀i 0≤i<k 
        i+=1  
    try:
        final_centroids = spkmeans.fit(df, C.tolist(), d, -1) 
    except:
        raise Exception("An Error Has Occurred")
    return [C_indices]+final_centroids
  


######################## tester #############################

dict = {"jacobi":0, "wam":1, "ddg":2, "lnorm":3, "spk":-2} # "T":4, "kmeans":-1
for goal in ["jacobi", "wam", "ddg", "lnorm", "spk"] :
    if goal=="jacobi":  f ="jacobi" 
    else: f = "spk"
    for i in range(20):
        if(i>9 and goal!="jacobi"):
            continue
        else:
            df=pd.read_csv("input/{}_{}.txt".format(f,i), header=None)
            N=df.shape[0]
            d=df.shape[1]
            df = df.to_numpy().tolist() 
            try:
                final_matrix = spkmeans.fit(df, None, d, dict[goal])# exe spkmeans on the data
            except:
                raise Exception("An Error Has Occurred")
            df=pd.read_csv("output/py/"+goal+"/"+"{}_{}.txt".format(f,i), header=None).to_numpy()
            if(goal=="spk"):
                k=df.shape[1]
                final_matrix = np.transpose(np.array(final_matrix)).tolist()
                quicksort(final_matrix, 0, len(final_matrix)-1)
                final_matrix = np.transpose(np.array(final_matrix))
                T = spkmeans.fit(final_matrix[1:,:k].tolist(), None, k, 4)
                final_matrix=kmeans_pp(T, N, k, k)
            x = 1 - abs(final_matrix-df)<0.0001   
            print("******* "+ goal.upper()+" : {}_{}.txt".format(f,i)+" *********")
            print("errors: " + str(np.sum(x)))
            print(~x)
       


    