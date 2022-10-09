# Face-Recognition
_Undergrad project made in Pattern Recognition course._

* The project was built using PCA. PCA is a statistical approach used for reducing the number of variables in face recognition. In PCA, every image in the training set is represented as a linear combination of weighted eigenvectors.
## Steps:
1. Dataset was downloaded as a zip file and downloaded to the drive.
2. Images were stacked into a matrix with a size of (400,10304) with a label vector y
of size (400,)
3. The matrix was then sliced into training and testing sets (50% train, 50% test)
Odd rows for training, even rows for testing→ giving per person 5 instances for
training and 5 instances for testing. Labels were split accordingly
4. PCA function was made from scratch
PCA Algorithm (DataMatrix) # DataMatrix here is the training set
compute mean for DataMatrix
Center the data (get Z)
Covariance Matrix
Eigenvalues and EigenVectors
5. A function to get the fraction of total variance according to different Alphas and
get the new eigen vectors to project on
6. Get different eigen vectors according to different alpha values
7. Compute mean for test data and center it
8. Project the training set and test sets separately using same projection matrix
9. Use a simple classifier (KNN) and compute accuracy
Plot alpha and the classification accuracy
