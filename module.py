import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def hostnames_list(filename):
    hostnames=[]
    with open(filename, 'r') as file:
        for line in file:
            line=line.split()
            hostnames.append(line[1])
    return hostnames

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
        else:
            labels.append(2)
    return np.array(labels),np.array(labeled_dataset)

def read_graph(filename,size):
    outlinks=[]
    i=0
    with open(filename, 'r') as file:
        mat=sparse.lil_matrix((size+1,size+1))
        for line in file:
            line=line.split()
            line=line[2:]
            l=len(line)
            if(l>0):
                for outlink in line:
                    outlink=outlink.split(':')
                    j=int(outlink[0])
                    mat[i,j]=int(outlink[1])
            else:
                mat[i,-1]=1         
            i+=1
    mat[-1,-1]=1
    mat=normalize(mat, norm='l1',axis=1)
    return mat.tocsr()

def PR_iteration(old_pr,R,n,alpha):
    P=(1-alpha)*R.T #allocations reduced and scipy code is used
    new_pr=alpha/n*np.ones(n)+P.dot(old_pr)#normalization choice: 1 (probability distribution)
    return new_pr

def compute_PR(alpha,epsilon,R):
    n=R.get_shape()[0]
    x=np.random.rand(n)
    x/=x.sum()
    err=np.inf
    while(err>epsilon):
        x_new=PR_iteration(x,R,n,alpha)
        err=(abs(x_new-x)).sum()
        print("Error:%.2E"%err,end='\r')
        x=x_new
    print("PageRank computed")
    return np.squeeze(np.asarray(x))

def columns_list(R):
    columns=[]
    n=R.get_shape()[0]
    for i in range(n):
        o=np.zeros(n)
        o[i]=1
        columns.append(R.dot(o))
    return columns

def pushback(u, p, r, alpha, R):
    p[u]+=alpha*r[u]
    column=R[u]
    r+=(1-alpha)*r[u]*column
    r[u]=0
    return p,r

def approximate_contributions(v, alpha, eps, pmax, R):
    n=len(R)
    p=np.zeros(n)
    r=np.zeros(n)
    r[v]=1/n
    u=v
    while True:
        p,r=pushback(u, p, r, alpha, R)
        u=np.argmax(r)        
        if r[u]<eps or np.linalg.norm(p, 1)>=pmax:
            break
    return p

def extract_features(R,delta,contributions,labeled_dataset,rank):
    supporting_set_size=np.zeros_like(labeled_dataset)
    contribution_from_supporting_set=np.zeros_like(labeled_dataset, dtype=np.float64)
    l2_norm=np.zeros_like(labeled_dataset,dtype=np.float64)
    indegree=np.zeros_like(labeled_dataset)
    pr_indegree=np.zeros_like(labeled_dataset,dtype=np.float64)
    outdegree=np.zeros_like(labeled_dataset)
    reciprocity=np.zeros_like(labeled_dataset,dtype=np.float64)  
    for i in range(len(labeled_dataset)):
        supporting_set=np.where(contributions[i]>delta*rank[labeled_dataset[i]])[0]
        supporting_set_size[i]=len(supporting_set)
        contribution_from_supporting_set[i]= contributions[i][supporting_set].sum()/rank[labeled_dataset[i]]
        l2_norm[i]=np.linalg.norm(contributions[i][supporting_set]/rank[labeled_dataset[i]], 2)
        inlinks=R[:,labeled_dataset[i]].nonzero()[0]
        outlinks=R[labeled_dataset[i],:].nonzero()[1]
        indegree[i]=(len(inlinks))
        outdegree[i]=(len(outlinks))
        reciprocity[i]=len(np.intersect1d(inlinks, outlinks))/outdegree[i]
        if indegree[i]>0:
            pr_indegree[i]=rank[labeled_dataset[i]]/indegree[i]
        else:
            pr_indegree[i]=1
    x=np.zeros((len(labeled_dataset),8))
    x[:,1]=supporting_set_size
    x[:,0]=contribution_from_supporting_set
    x[:,2]=l2_norm
    x[:,3]=indegree
    x[:,4]=outdegree
    x[:,5]=rank[labeled_dataset]
    x[:,6]=pr_indegree
    x[:,7]=reciprocity        
    return x


def top_n_percent(n,rank,labeled_dataset):
    indices_top = (-rank).argsort()[:(n*len(rank))//100]
    labeled_top=[]
    for i in range(len(labeled_dataset)):
        if labeled_dataset[i] in indices_top:
            labeled_top.append(i)
    return np.array(labeled_top)