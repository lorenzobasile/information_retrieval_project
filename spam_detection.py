import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall

'''
hostnames_array: the file containing the names of hosts is read and an array containing the names is returned.
'''

def hostnames_array(filename):
    hostnames=[]
    with open(filename, 'r') as file:
        for line in file:
            line=line.split()
            hostnames.append(line[1])
    return np.array(hostnames)

'''
labels_dictionary reads the file containing labels for some of the hosts and returns a dictionary whose keys are hostnames 
and whose corresponding values are their labels, encoded as 0 for normal (nonspam) hosts and 1 for spam.
'''

def labels_dictionary(filename):
    labels={}
    with open(filename, 'r') as file:
        for line in file:
            line=line.split()
            if line[3]=='normal':
                labels[line[0]]=0
            elif line[3]=='spam':
                labels[line[0]]=1
    return labels

'''
make_dataset takes as input an array of hostnames and a dictionary matching hostnames to labels. The return values are 
an array containing the labels of the labeled dataset and an array containing the indices of the labeled samples.
'''

def make_dataset(labels_dict, hostnames_list):
    labels=[]
    labeled_dataset=[]
    for i in range(len(hostnames_list)):
        label=labels_dict.get(hostnames_list[i])
        if label==0:
            labeled_dataset.append(i)
            labels.append(0)
        elif label==1:
            labeled_dataset.append(i)
            labels.append(1)
    return np.array(labels),np.array(labeled_dataset)

'''
read_graph is used to read the web graph provided with the dataset. The returned value is the transpose of a scipy csc sparse matrix whose
i,j-th element is equal to 1 over the total number of outlinks of host i if there is a link from i to j, 0 otherwise.
The returned matrix has to be column-stochastic so if a node has no outlinks it is linked to an artificial node provided with a self-loop.
'''

def read_graph(filename,size):
    outlinks=[]
    i=0
    with open(filename, 'r') as file:
        mat=sparse.lil_matrix((size+1, size+1))
        for line in file:
            line=line.split()
            line=line[2:]
            l=len(line)
            if(l>0):
                for outlink in line:
                    outlink=outlink.split(':')
                    j=int(outlink[0])
                    mat[i, j]=1/l
            else:
                mat[i, -1]=1         
            i+=1
    mat[-1, -1]=1
    return mat.tocsc().T

'''
compute_PR performs the PageRank computation. The input parameters are alpha (the teleporting constant), epsilon (the precision of the
iterative computation) and R_T (transpose of the transition matrix).
The return value is x, which is initialized at random and then iteratively updated through PageRank iterations up to a precision of epsilon.
'''

def compute_PR(alpha, epsilon, R_T):
    n=R_T.get_shape()[0]
    x=np.random.rand(n)
    x/=x.sum()
    err=np.inf
    while(err>epsilon):
        x_new=alpha/n*np.ones(n)+(1-alpha)*R_T.dot(x)
        err=(abs(x_new-x)).sum()
        print("Error:%.2E"%err, end='\r')
        x=x_new
    print("PageRank computed")
    return np.squeeze(np.asarray(x))


'''
pushback function (as presented in Andersen et al., 2007).
'''

def pushback(u, p, r, alpha, R_T):
    p[u]+=alpha*r[u]
    row=R_T[u]
    r+=(1-alpha)*r[u]*row
    r[u]=0
    return p,r

'''
approximate_contributions function (as presented in Andersen et al., 2007).
'''

def approximate_contributions(v, alpha, eps, pmax, R_T):
    n=len(R_T)
    p=np.zeros(n)
    r=np.zeros(n)
    r[v]=1/n
    u=v
    norm=0
    while True:
        norm+=alpha*r[u]
        p_prime,r=pushback(u, p, r, alpha, R_T)
        u=np.argmax(r)
        if norm>=pmax:
            break
        p=p_prime    
        if r[u]<=eps:
            break
    return p

'''
extract_features computes and returns the set of features to be used for spam/non spam classification.
'''

def extract_features(R_T, delta, contributions, labeled_dataset, rank):
    supporting_set_size=np.zeros_like(labeled_dataset)
    contribution_from_supporting_set=np.zeros_like(labeled_dataset, dtype=np.float64)
    l2_norm=np.zeros_like(labeled_dataset, dtype=np.float64)
    indegree=np.count_nonzero(R_T, axis=1)
    outdegree=np.count_nonzero(R_T, axis=0)
    for i in range(len(labeled_dataset)):
        supporting_set=np.where(contributions[i]>delta*rank[labeled_dataset[i]])[0]
        supporting_set_size[i]=len(supporting_set)
        contribution_from_supporting_set[i]= contributions[i][supporting_set].sum()/rank[labeled_dataset[i]]
        l2_norm[i]=np.linalg.norm(contributions[i][supporting_set]/rank[labeled_dataset[i]], 2)
    return indegree[labeled_dataset], outdegree[labeled_dataset], supporting_set_size, contribution_from_supporting_set, l2_norm


'''
print_prediction_metrics takes a classifier, a set of features, their corresponding labels and the number of folds to perform CV with.
after a call to cross_val_predict of sklearn, 3 metrics are printed: accuracy, precision on spam class, recall on spam class.
'''

def print_prediction_metrics(clf, x, y, k):
    pred=cross_val_predict(clf, x, y, cv=StratifiedKFold(n_splits=k, shuffle=True))
    print("Accuracy: ", round(accuracy(y, pred), 2))
    print("Precision on spam: ", round(precision(y, pred, average=None)[1], 3))
    print("Recall on spam: ", round(recall(y, pred, average=None)[1], 3))
    return