import numpy as np, collections as coll, matplotlib.pyplot as plt, itertools as itl, time

#Part 1.1

iris_data  = np.loadtxt('fisheriris.data', delimiter=",", unpack=False)

feature_names = np.genfromtxt('attributes.txt',  delimiter=",", dtype='U')
feature_names = feature_names.tolist() 

target_names = np.genfromtxt('feature.txt',  delimiter=",", dtype='U')
target_names = target_names.tolist()

#Part 1.2
target = np.int16(iris_data[:,4]) 


#np.s[n] => where n is the index to be deleted axis = 0 is row and 1 is column
#np.s[f:u] => where f is the index from where the deletion will start and u is the index before which the deletion will be done
iris_data = np.delete(iris_data, np.s_[4], axis = 1)

feature_names.remove('class')

##PRINT AFTER REMOVING COLUMNS 
#** WHAT IS FIND THE SIZE OF FISHERS MEASUREMENT**
iris_data_size = iris_data.size
print("Size of Fisher's measurement:\n" + str (iris_data_size))

iris_data_dim2 = iris_data.shape
print("Number of elements of second dimension: \n" + str(iris_data_dim2[1]))

#Part 1.3
#>>Sum of each columns
iris_sum_cols = iris_data.sum(axis = 0)
print("\nSum of first column:\n " + str(iris_sum_cols[0]))
print("\nSum of second column:\n " + str(iris_sum_cols[1]))
print("\nSum of third column:\n " + str(iris_sum_cols[2]))
print("\nSum of fourth column:\n " + str(iris_sum_cols[3]))

#>>Sum of just second and fourth column //Re confirm the meaning of question
iris_sum_col24 =  iris_sum_cols[1] + iris_sum_cols[3]
print("\nSum of second and fourth columns:\n " + str(iris_sum_col24))

#>>Pull sample rows from 27 through 48 and find max value in each of four col 
#>>of this set
iris_data_2748 = iris_data[26:47]
print("\nRows 27 through 48:" + str(iris_data_2748))

iris_data_2748Max = np.amax(iris_data_2748, axis = 1)
print("\nMax of each Column from 27 through 48:\n " + str(iris_data_2748Max))

#>>Pull odd between 3 and 33 and find min from first 2 columns
iris_data_333Odd = iris_data[2:32:3] #getting from 3rd row upto 33rd row and picking every 3rd/odd element
print("\nRows 3 to 33 that are odd: " + str(iris_data_333Odd))
iris_data_333Odd_12Col = np.delete(iris_data_333Odd, np.s_[2:4], axis = 1)
iris_data_333Odd_12ColMin = np.amin(iris_data_333Odd_12Col, axis = 1)
print("\nMin from 3 to 33 odd, first and second column:\n" + str(iris_data_333Odd_12ColMin))


#Part 1.4
#>>Sum of 1 and 3 col, and store it in r13rd
iris_data_Col13 = np.delete(iris_data,np.s_[1:4:2], axis = 1)
r13rd = np.sum(iris_data_Col13, axis = 1);
print("\nSum of 1st and 3rd column :\n " + str(r13rd))

#>>Cubed of elements
iris_data_Col13_cube = np.power(r13rd , 3)
print("\nCube of 1st and 3rd column elements:\n " + str(iris_data_Col13_cube)) 


#Part 1.5
#>>Extract first 4 rows of first 2 columns
mat1 = np.delete(iris_data[0:4],np.s_[2:4], axis = 1)
print("\nFirst 4 rows of first 2 columns:\n " + str(mat1))

#>>Extract first 4 rows of last 2 column
mat2 = np.delete(iris_data[0:4],np.s_[0:2], axis = 1)
print("\nFirst 4 rows of Last 2 columns:\n " + str(mat2))

#>>Add mat1 and mat2
mat1_Matrix = np.matrix(mat1)
mat2_Matrix = np.matrix(mat2)
sum_mat1_mat2 = mat1_Matrix + mat2_Matrix 
print("\nSum of mat1 and mat2:\n" + str(sum_mat1_mat2))

#>>Multiply mat1 and mat2
mat1_squeeze = np.squeeze(np.asarray(mat1_Matrix))
mat2_squeeze = np.squeeze(np.asarray(mat2_Matrix))
mul_mat1_mat2 = mat1_squeeze * mat2_squeeze
print("\nMultiplication of elements in mat1 and mat2:\n" + str(mul_mat1_mat2))

#Part 1.6 ||Did solve it, but need to understand why dot vs inner
mat3 = np.inner(mat1_Matrix, mat2_Matrix)
print("\nInner product of mat1 and mat2:\n" + str(mat3))

np.savetxt('mat3.CSV',mat3, delimiter=',')

#Part 1.7 
#>>Plot mean and standard deviation
iris_data_mean = np.mean(iris_data, axis = 1)
print("\nMean of iris data:\n" + str(iris_data_mean))
iris_data_std = np.std(iris_data, axis = 1)
print("\nStd of iris data:\n" + str(iris_data_mean))
#print("xdata:" + str(np.arange(iris_data[0])) + "ydata:" + str(len(iris_data_mean[0])) )
#import pdb; pdb.set_trace()
plt.errorbar(np.arange(len(iris_data)), iris_data_mean , yerr=iris_data_std, fmt='ok', lw=3)
plt.title('Mean and standard deviation of Fishers measurement')
plt.xlabel('Fishers measurements')
#>>Verify if correct
plt.ylabel('Standard devitation')
plt.show()

#Part 1.9
iris_data_var1_x = np.float64(iris_data[:,0]) 
iris_data_var2_y = np.float64(iris_data[:,1]) 
#target_colors = [target_string[i].replace('0','g').replace('1','r').replace('2','b') for i in range(len(target_string))]

target_color_list = []
target_distinct_color = []
target_distinct_legend = []
for target_per in target:
    if target_per == 0:
        target_color_list.append('r') 
        if 'r' not in target_distinct_color:
            target_distinct_color.append('r')
            target_distinct_legend.append(plt.Rectangle((0, 0), 1, 1, fc='r'))
    elif target_per == 1:
        target_color_list.append('g')
        if 'g' not in target_distinct_color:
            target_distinct_color.append('g')
            target_distinct_legend.append(plt.Rectangle((0, 0), 1, 1, fc='g'))
    elif target_per == 2:
        target_color_list.append('b')
        if 'b' not in target_distinct_color:
            target_distinct_color.append('b')
            target_distinct_legend.append(plt.Rectangle((0, 0), 1, 1, fc='b'))
#print('target colors' + str(target_string))
#print("Distinct colors: " + str(target_distinct_color))
plt.scatter(iris_data_var1_x,iris_data_var2_y, c=target_color_list)
plt.title('Distribution of Fishers Measurement as per the classes')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(target_distinct_legend,target_names)
plt.show()


#Part 1.9
#iris_data_comb = itl.combinations(iris_data,2)
iris_data_com = [None] * 6
iris_data_com[0]  = []
iris_data_com[1]  = []
iris_data_com[2]  = []
iris_data_com[3]  = []
iris_data_com[4]  = []
iris_data_com[5]  = []

for i in range(len(iris_data)):
    j = 1
    for n in itl.combinations(iris_data[i],2):
        iris_data_com[j-1].append(n)
        #print ("After:" + str(j),"[:]",iris_data_com[j-1])
        j += 1
        
#print ("\n\nLength:" + str(len(iris_data_com[0])) +"\n\n"+str(0),"[:]",iris_data_com[0])
#print ("\n\nLength:" + str(len(iris_data_com[1])) +"\n\n" + str(1),"[:]",iris_data_com[1])
#print ("\n\nLength:" + str(len(iris_data_com[2])) +"\n\n" + str(2),"[:]",iris_data_com[2])
#print ("\n\nLength:" + str(len(iris_data_com[3])) +"\n\n" + str(3),"[:]",iris_data_com[3])
#print ("\n\nLength:" + str(len(iris_data_com[4])) +"\n\n" + str(4),"[:]",iris_data_com[4])
#print ("\n\nLength:" + str(len(iris_data_com[5])) +"\n\n" + str(5),"[:]",iris_data_com[5])

for i in range(len(iris_data_com)):
    iris_data_var1_x = np.float64(iris_data_com[i])[:,0] 
    iris_data_var2_y = np.float64(iris_data_com[i])[:,1] 
    plt.scatter(iris_data_var1_x,iris_data_var2_y, c='r') 
    plt.show()
    
#Part 1.10
#randon state is more like a class that can be used to invoke method (Reinitializes the existing instance)
#eg r = np.random.RandomState(1234)
# r.uniform(0,10,5)
#array([ 1.9151945 ,  6.22108771,  4.37727739,  7.85358584,  7.79975808])
#random seed is more like a fully qualified namespace in C#, you use it along with np(creates a new instance)
#np.random.seed(1234)
#np.random.uniform(0, 10, 5)
#array([ 1.9151945 ,  6.22108771,  4.37727739,  7.85358584,  7.79975808])
for i in range(3):
   np.random.seed()
   cent = np.random.randn(5,5)
   print("cent matrix:\n", np.matrix(cent))

   #>>Get the fix pseudo number part figured out
   ran_num = np.random.randint(10)
   r = np.random.RandomState(ran_num)
   cent_fix = r.randn(5,5)
   print("cent fix matrix:\n", np.matrix(cent_fix)) 
   
a = np.array([[1, 2]])
b = np.array([[5, 6]])
np.hstack((a,b))
   
#Part 1.11
v = np.random.uniform(size = 5)
v_add = np.random.uniform(size = 5)

v = np.hstack((v,v_add))
print("Concatenated v variable: \n", v)

