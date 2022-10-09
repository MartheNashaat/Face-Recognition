# **Open Zip file and open images**

---
"""

import zipfile
from PIL import Image
import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt

#Open the zip file
ReadZip = zipfile.ZipFile('/content/drive/MyDrive/Colab Notebooks/archivee.zip', mode='r')
NameList=ReadZip.namelist() #Name list of elements
AllImages=NameList[1:] #remove readme from list
dataMatrix=np.empty((0,10304),int) #empty matrix

for i in AllImages:
    myfile = ReadZip.open(i) #open file
    img = Image.open(myfile) #get image to display as image
    image_sequence = img.getdata()
    image_array = np.int8(np.array(image_sequence)) #convert hex to int 8
    dataMatrix=np.vstack([dataMatrix,image_array])
#data matrix of size (400,10304)
labels=np.repeat(np.arange(1,41),10) #size=400
print(dataMatrix.shape)
ReadZip.close()# Close the archive releasing it from memory

"""# **Train and test**

---


"""

#odd rows for training and even for testing from -->datamatrix
TestingData,TrainingData=dataMatrix[0::2],dataMatrix[1::2] # slice [start:end:step]
TrainLabel=TestLabel=labels[::2] #5 for train , 5 for test #size=200
print(dataMatrix.shape)

"""# **PCA function**

---



"""

def PCA(data): #data= matrix, ex: training data
  meann=np.mean(data,axis=0)
  z=data-meann #centered
  covMatrix=np.cov(z,rowvar=False,bias=True)
  #Eigen values and eigen vectors
  Eigenvalues,Eigenvectors= np.linalg.eigh(covMatrix)
  return meann,z,Eigenvalues,Eigenvectors

Trainmean,TrainZ,Eigenvalues,Eigenvectors=PCA(TrainingData)
  sortedindex=np.argsort(Eigenvalues)
  sortedEigenvalue =np.sort(Eigenvalues)
  sortedEigenvectors = Eigenvectors[:,sortedindex]

#fraction of total variance 
#eigenvalue of r / sum of eigen values where r= 1 , 1 2, 1 2 3, .... d
def projectMatrixU(sortedEigenvalue,alpha):
  accEigenvalues=0
  fracTotalVar=0
  for i in sortedEigenvalue:
    accEigenvalues+=i
    fracTotalVar= accEigenvalues/np.sum(sortedEigenvalue)
    if fracTotalVar >= alpha:
      index=int(np.where(sortedEigenvalue==i)[0])
      break
  sortedEigVectnew=sortedEigenvectors[:,0:index]
  return sortedEigVectnew

#get dimensions of 4 alphas(0.8,0.85,0.9,0.95)
#dimension of alpha == sorted eigenvect new 
dimensionsofAlpha1=projectMatrixU(sortedEigenvalue,0.8)
dimensionsofAlpha2=projectMatrixU(sortedEigenvalue,0.85)
dimensionsofAlpha3=projectMatrixU(sortedEigenvalue,0.9)
dimensionsofAlpha4=projectMatrixU(sortedEigenvalue,0.95)

#center test data
def TestData(testdata):
  Testmean= np.mean(testdata,axis=0)
  TestZ=testdata-Testmean #centered
  return Testmean,TestZ

"""Get new dimensions and project"""

Testmean,TestZ=TestData(TestingData)
def projectDataNewdim(matrix,TrainZ,TestZ): #matrix represents dimensions of alpha= sortedEigenvectnew
  trainProjected=np.dot(matrix.T,TrainZ.T).T
  testProjected=np.dot(matrix.T,TestZ.T).T
  return trainProjected, testProjected

#project train and test data on new reduced dimensions
reducedDimTrain1,reducedDimTest1=projectDataNewdim(dimensionsofAlpha1,TrainZ,TestZ)
reducedDimTrain2,reducedDimTest2=projectDataNewdim(dimensionsofAlpha2,TrainZ,TestZ)
reducedDimTrain3,reducedDimTest3=projectDataNewdim(dimensionsofAlpha3,TrainZ,TestZ)
reducedDimTrain4,reducedDimTest4=projectDataNewdim(dimensionsofAlpha4,TrainZ,TestZ)

"""# **Classifier**

---


"""

#find nearest neighbors to detect class label using knn from sklearn
from sklearn.neighbors import KNeighborsClassifier as knnClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics #to check accuracy

def kNearestNeighbor(Traindata,Testdata,TrainLabel,TestLabel,k):
  ConfusionMt300=[]
  knn=knnClassifier(n_neighbors=k)
  knn.fit(Traindata,TrainLabel)
  predict=knn.predict(Testdata)
  ConfusionMt300=confusion_matrix(TestLabel,predict )
  print(ConfusionMt300)
  accuracy=metrics.accuracy_score(TestLabel,predict)
  return accuracy

Acc1=kNearestNeighbor(reducedDimTrain1,reducedDimTest1,TrainLabel,TestLabel,1) #0.8

Acc2=kNearestNeighbor(reducedDimTrain2,reducedDimTest2,TrainLabel,TestLabel,1) #0.85

Acc3=kNearestNeighbor(reducedDimTrain3,reducedDimTest3,TrainLabel,TestLabel,1) #0.9

Acc4=kNearestNeighbor(reducedDimTrain4,reducedDimTest4,TrainLabel,TestLabel,1) #0.95

#plot accuracy with alpha values
plt.plot([0.8,0.85,0.9,0.95],[Acc1,Acc2,Acc3,Acc4])

#classifier tuning using PCA 
PCAaccuracy=[]
for k in range(1,8,2):
  PCAaccuracy.append(kNearestNeighbor(reducedDimTrain1,reducedDimTest1,TrainLabel,TestLabel,k))
plt.plot([1,3,5,7],PCAaccuracy)

"""# **Bonus Part**
Training = 7 instances per subject
Testing = 3 instances per subject

---
"""

#odd rows for training and even for testing from -->datamatrix
splittedMatrix=np.array_split(dataMatrix,40)
splittedMatrix=np.array(splittedMatrix)
print(splittedMatrix.shape)
TrainingBonus=np.empty((0,10304),int)
TestingBonus=np.empty((0,10304),int)
for i in splittedMatrix:
  TrainingDataBonus=i[:7:]
  TestingDataBonus=i[7::]
  TrainingBonus=np.vstack([TrainingBonus,TrainingDataBonus])
  TestingBonus=np.vstack([TestingBonus,TestingDataBonus])
print(TrainingBonus.shape)
print(TestingBonus.shape)

#bonusLabels=np.repeat(np.arange(1,41),10)
TrainingBonusLabel=[]
TestingBonusLabel=[]
for i in range(1,41):
    TrainingBonusLabel=np.append(TrainingBonusLabel,np.repeat(i,7))
    TestingBonusLabel=np.append(TestingBonusLabel,np.repeat(i,3))

"""# **PCA for Bonus -split( 70%-30% )**

---

"""

TrainmeanBonus,TrainZBonus,EigenvaluesBonus,EigenvectorsBonus=PCA(TrainingBonus)
sortedindexB=np.argsort(EigenvaluesBonus)
sortedEigenvalueB =np.sort(EigenvaluesBonus)
sortedEigenvectorsB = EigenvectorsBonus[:,sortedindexB]

#projectMatrixU(sortedEigenvalue,alpha): bt return sorted eigenvectors new 
dimensionsofAlpha1Bonus=projectMatrixU(sortedEigenvalueB,0.8)
dimensionsofAlpha2Bonus=projectMatrixU(sortedEigenvalueB,0.85)
dimensionsofAlpha3Bonus=projectMatrixU(sortedEigenvalueB,0.9)
dimensionsofAlpha4Bonus=projectMatrixU(sortedEigenvalueB,0.95)

TestmeanB,TestZbonus=TestData(TestingBonus)
reducedDimTrain1Bonus,reducedDimTest1Bonus=projectDataNewdim(dimensionsofAlpha1Bonus,TrainZBonus,TestZbonus)
reducedDimTrain2Bonus,reducedDimTest2Bonus=projectDataNewdim(dimensionsofAlpha2Bonus,TrainZBonus,TestZbonus)
reducedDimTrain3Bonus,reducedDimTest3Bonus=projectDataNewdim(dimensionsofAlpha3Bonus,TrainZBonus,TestZbonus)
reducedDimTrain4Bonus,reducedDimTest4Bonus=projectDataNewdim(dimensionsofAlpha4Bonus,TrainZBonus,TestZbonus)
print(reducedDimTrain1Bonus.shape)

"""# **Comparing Accuracy of different splits**

---


"""

AccuracyBonus1=kNearestNeighbor(reducedDimTrain1Bonus,reducedDimTest1Bonus,TrainingBonusLabel,TestingBonusLabel,1)

AccuracyBonus2=kNearestNeighbor(reducedDimTrain2Bonus,reducedDimTest2Bonus,TrainingBonusLabel,TestingBonusLabel,1)

AccuracyBonus3=kNearestNeighbor(reducedDimTrain3Bonus,reducedDimTest3Bonus,TrainingBonusLabel,TestingBonusLabel,1)

AccuracyBonus4=kNearestNeighbor(reducedDimTrain4Bonus,reducedDimTest4Bonus,TrainingBonusLabel,TestingBonusLabel,1)

plt.plot([0.8,0.85,0.9,0.95],[AccuracyBonus1,AccuracyBonus2,AccuracyBonus3,AccuracyBonus4])

"""# **Faces=200,NonFaces=200**

---


"""

#take 200 from faces 
ReadZip = zipfile.ZipFile('/content/drive/MyDrive/Colab Notebooks/archivee.zip', mode='r')
AllImagesFaces=NameList[1:201] #remove readme from list
FacesdataMatrix=np.empty((0,10304),int) #empty matrix 
for i in AllImagesFaces:
    Fmyfile = ReadZip.open(i) #open file
    Fimg = Image.open(Fmyfile) #get image to display as image
    Fimage_sequence = Fimg.getdata()
    Fimage_array = np.int8(np.array(Fimage_sequence)) #convert hex to int 8
   # print(Fimage_array.shape)
    FacesdataMatrix=np.vstack([FacesdataMatrix,Fimage_array])
#data matrix of size (400,10304)
#Face labels = 'Faces'
Facelabels=np.repeat('Faces',200) #size=200

#Open the zip file
NonFacesBonus = zipfile.ZipFile('/content/drive/MyDrive/Colab Notebooks/nonfaces.zip', mode='r')
NonFaceNameList=NonFacesBonus.namelist() #Name list of elements
NonFaceImages200=NonFaceNameList[1:201] #remove readme from list and take first 200 images
NonFacesdataMatrix200=np.empty((0,10304),int) #empty matrix
newsize=(92,112)

for j in NonFaceImages200:
    nfmyfile = NonFacesBonus.open(j) #open file
    nfimg = Image.open(nfmyfile) #get image to display as 
    nfimg=nfimg.resize(newsize)
    nfimage_sequence = nfimg.getdata()
    nfimage_array = np.int8(np.array(nfimage_sequence)) #convert hex to int 8
    nfimage_array=nfimage_array.flatten()
    NonFacesdataMatrix200=np.vstack([NonFacesdataMatrix200,nfimage_array[:10304]])
    #display(nfimage_array)
#NonFacesBonus.close()# Close the archive releasing it from memory

NonFacelabels200=np.repeat('NonFaces',200) #size=200

BonusdataMatrix200= np.empty((0,10304),int)
FaceNonFaceLabels200=np.empty(0,int)
BonusdataMatrix200=np.concatenate((BonusdataMatrix200,FacesdataMatrix,NonFacesdataMatrix200))
FaceNonFaceLabels200=np.concatenate((FaceNonFaceLabels200,Facelabels,NonFacelabels200))
print(BonusdataMatrix200.shape)
print(FaceNonFaceLabels200.shape)

#odd rows for training and even for testing from -->datamatrix
TestingDataFNBonus,TrainingDataFNBonus=BonusdataMatrix200[0::2],BonusdataMatrix200[1::2] # slice [start:end:step]
TrainLabelFNBonus=TestLabelFNBonus=FaceNonFaceLabels200[::2] #5 for train , 5 for test #size=200

MeanFN,ZFN,EigenvaluesFN,EigenvectorsFN=PCA(TrainingDataFNBonus)

sortedindexFN=np.argsort(EigenvaluesFN)
sortedEigenvalueFN =np.sort(EigenvaluesFN)
sortedEigenvectorsFN = EigenvectorsFN[:,sortedindexFN]

#get dimensions of 4 alphas(0.8,0.85,0.9,0.95)
#dimension of alpha == sorted eigenvect new 
dimensionsofAlpha1FN=projectMatrixU(sortedEigenvalueFN,0.8)
dimensionsofAlpha2FN=projectMatrixU(sortedEigenvalueFN,0.85)
dimensionsofAlpha3FN=projectMatrixU(sortedEigenvalueFN,0.9)
dimensionsofAlpha4FN=projectMatrixU(sortedEigenvalueFN,0.95)

TestmeanFN,TestZFN=TestData(TestingDataFNBonus)

#project train and test data on new reduced dimensions
reducedDimTrain1FN,reducedDimTest1FN=projectDataNewdim(dimensionsofAlpha1FN,ZFN,TestZFN) #Alpha 0.8
reducedDimTrain2FN,reducedDimTest2FN=projectDataNewdim(dimensionsofAlpha2FN,ZFN,TestZFN)
reducedDimTrain3FN,reducedDimTest3FN=projectDataNewdim(dimensionsofAlpha3FN,ZFN,TestZFN)
reducedDimTrain4FN,reducedDimTest4FN=projectDataNewdim(dimensionsofAlpha4FN,ZFN,TestZFN)

AccuracyBonusFN1=kNearestNeighbor(reducedDimTrain1FN,reducedDimTest1FN,TrainLabelFNBonus,TestLabelFNBonus,1)

AccuracyBonusFN2=kNearestNeighbor(reducedDimTrain2FN,reducedDimTest2FN,TrainLabelFNBonus,TestLabelFNBonus,1)

AccuracyBonusFN3=kNearestNeighbor(reducedDimTrain3FN,reducedDimTest3FN,TrainLabelFNBonus,TestLabelFNBonus,1)

AccuracyBonusFN4=kNearestNeighbor(reducedDimTrain4FN,reducedDimTest4FN,TrainLabelFNBonus,TestLabelFNBonus,1)

# Plot the accuracy vs the number of non-faces images while fixing the number of face images.
#face images=200 , nonfaces=200,300,400
plt.plot([200,300,400],[AccuracyBonusFN1,AccuracyBonusFN1_300,AccuracyBonusFN1_400])

"""# **Faces=200, Non Faces=300**

---


"""

NonFaceImages300=NonFaceNameList[1:301] #remove readme from list and take first 300 images
NonFacesdataMatrix300=np.empty((0,10304),int) #empty matrix

for j in NonFaceImages300:
    nnfmyfile = NonFacesBonus.open(j) #open file
    nnfimg = Image.open(nnfmyfile) #get image to display as 
    nnfimg=nfimg.resize(newsize)
    nnfimage_sequence = nnfimg.getdata()
    nnfimage_array = np.int8(np.array(nnfimage_sequence)) #convert hex to int 8
    nnfimage_array=nnfimage_array.flatten()
    NonFacesdataMatrix300=np.vstack([NonFacesdataMatrix300,nnfimage_array[:10304]])
    #display(nfimage_array)
  

#NonFacesBonus.close()# Close the archive releasing it from memory

NonFacelabels300=np.repeat('NonFaces',300) #size=300

BonusdataMatrix300= np.empty((0,10304),int)
FaceNonFaceLabels300=np.empty(0,int)
BonusdataMatrix300=np.concatenate((BonusdataMatrix300,FacesdataMatrix,NonFacesdataMatrix300))
FaceNonFaceLabels300=np.concatenate((FaceNonFaceLabels300,Facelabels,NonFacelabels300))
print(BonusdataMatrix300.shape)
print(FaceNonFaceLabels300.shape)

TestingDataFNBonus300,TrainingDataFNBonus300=BonusdataMatrix300[0::2],BonusdataMatrix300[1::2] # slice [start:end:step]
TrainLabelFNBonus300=TestLabelFNBonus300=FaceNonFaceLabels300[::2] #5 for train , 5 for test #size=200

MeanFN300,ZFN300,EigenvaluesFN300,EigenvectorsFN300=PCA(TrainingDataFNBonus300)

sortedindexFN300=np.argsort(EigenvaluesFN300)
sortedEigenvalueFN300 =np.sort(EigenvaluesFN300)
sortedEigenvectorsFN300 = EigenvectorsFN300[:,sortedindexFN300]

#get dimensions of 4 alphas(0.8,0.85,0.9,0.95)
#dimension of alpha == sorted eigenvect new 
dimensionsofAlpha1FN300=projectMatrixU(sortedEigenvalueFN300,0.8)
dimensionsofAlpha2FN300=projectMatrixU(sortedEigenvalueFN300,0.85)
dimensionsofAlpha3FN300=projectMatrixU(sortedEigenvalueFN300,0.9)
dimensionsofAlpha4FN300=projectMatrixU(sortedEigenvalueFN300,0.95)

TestmeanFN300,TestZFN300=TestData(TestingDataFNBonus300)

#project train and test data on new reduced dimensions
reducedDimTrain1FN300,reducedDimTest1FN300=projectDataNewdim(dimensionsofAlpha1FN300,ZFN300,TestZFN300) #Alpha =0.8
reducedDimTrain2FN300,reducedDimTest2FN300=projectDataNewdim(dimensionsofAlpha2FN300,ZFN300,TestZFN300)
reducedDimTrain3FN300,reducedDimTest3FN300=projectDataNewdim(dimensionsofAlpha3FN300,ZFN300,TestZFN300)
reducedDimTrain4FN300,reducedDimTest4FN300=projectDataNewdim(dimensionsofAlpha4FN300,ZFN300,TestZFN300)

AccuracyBonusFN1_300=kNearestNeighbor(reducedDimTrain1FN300,reducedDimTest1FN300,TrainLabelFNBonus300,TestLabelFNBonus300,1)

AccuracyBonusFN2_300=kNearestNeighbor(reducedDimTrain2FN300,reducedDimTest2FN300,TrainLabelFNBonus300,TestLabelFNBonus300,1)

AccuracyBonusFN3_300=kNearestNeighbor(reducedDimTrain3FN300,reducedDimTest3FN300,TrainLabelFNBonus300,TestLabelFNBonus300,1)

AccuracyBonusFN4_300=kNearestNeighbor(reducedDimTrain4FN300,reducedDimTest4FN300,TrainLabelFNBonus300,TestLabelFNBonus300,1)

"""# **Faces=200, Non Faces=400**

---
"""

NonFaceImages400=NonFaceNameList[1:401] #remove readme from list and take first 400 images
NonFacesdataMatrix400=np.empty((0,10304),int) #empty matrix
for j in NonFaceImages400:
    nnffmyfile = NonFacesBonus.open(j) #open file
    nnffimg = Image.open(nnffmyfile) #get image to display as 
    nnffimg=nfimg.resize(newsize)
    nnffimage_sequence = nnffimg.getdata()
    nnffimage_array = np.int8(np.array(nnffimage_sequence)) #convert hex to int 8
    nnffimage_array=nnffimage_array.flatten()
    NonFacesdataMatrix400=np.vstack([NonFacesdataMatrix400,nnffimage_array[:10304]])
    #display(nfimage_array)
  

#NonFacesBonus.close()# Close the archive releasing it from memory

NonFacelabels400=np.repeat('NonFaces',400) #size=400

BonusdataMatrix400= np.empty((0,10304),int)
FaceNonFaceLabels400=np.empty(0,int)
BonusdataMatrix400=np.concatenate((BonusdataMatrix400,FacesdataMatrix,NonFacesdataMatrix400))
FaceNonFaceLabels400=np.concatenate((FaceNonFaceLabels400,Facelabels,NonFacelabels400))
print(BonusdataMatrix400.shape)
print(NonFacesdataMatrix400.shape)

TestingDataFNBonus400,TrainingDataFNBonus400=BonusdataMatrix400[0::2],BonusdataMatrix400[1::2] # slice [start:end:step]
TrainLabelFNBonus400=TestLabelFNBonus400=FaceNonFaceLabels400[::2] #5 for train , 5 for test #size=200

MeanFN400,ZFN400,EigenvaluesFN400,EigenvectorsFN400=PCA(TrainingDataFNBonus400)

sortedindexFN400=np.argsort(EigenvaluesFN400)
sortedEigenvalueFN400 =np.sort(EigenvaluesFN400)
sortedEigenvectorsFN400 = EigenvectorsFN400[:,sortedindexFN400]

#get dimensions of 4 alphas(0.8,0.85,0.9,0.95)
#dimension of alpha == sorted eigenvect new 
dimensionsofAlpha1FN400=projectMatrixU(sortedEigenvalueFN400,0.8)
dimensionsofAlpha2FN400=projectMatrixU(sortedEigenvalueFN400,0.85)
dimensionsofAlpha3FN400=projectMatrixU(sortedEigenvalueFN400,0.9)
dimensionsofAlpha4FN400=projectMatrixU(sortedEigenvalueFN400,0.95)

TestmeanFN400,TestZFN400=TestData(TestingDataFNBonus400)

#project train and test data on new reduced dimensions
reducedDimTrain1FN400,reducedDimTest1FN400=projectDataNewdim(dimensionsofAlpha1FN400,ZFN400,TestZFN400) #Alpha =0.8
reducedDimTrain2FN400,reducedDimTest2FN400=projectDataNewdim(dimensionsofAlpha2FN400,ZFN400,TestZFN400)
reducedDimTrain3FN400,reducedDimTest3FN400=projectDataNewdim(dimensionsofAlpha3FN400,ZFN400,TestZFN400)
reducedDimTrain4FN400,reducedDimTest4FN400=projectDataNewdim(dimensionsofAlpha4FN400,ZFN400,TestZFN400)

AccuracyBonusFN1_400=kNearestNeighbor(reducedDimTrain1FN400,reducedDimTest1FN400,TrainLabelFNBonus400,TestLabelFNBonus400,1)

AccuracyBonusFN2_400=kNearestNeighbor(reducedDimTrain2FN400,reducedDimTest2FN400,TrainLabelFNBonus400,TestLabelFNBonus400,1)

AccuracyBonusFN3_400=kNearestNeighbor(reducedDimTrain3FN400,reducedDimTest3FN400,TrainLabelFNBonus400,TestLabelFNBonus400,1)

AccuracyBonusFN4_400=kNearestNeighbor(reducedDimTrain4FN400,reducedDimTest4FN400,TrainLabelFNBonus400,TestLabelFNBonus400,1)
