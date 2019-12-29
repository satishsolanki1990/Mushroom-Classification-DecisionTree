#import numpy as np
import pandas as pd

# node class
class CreateNode:
    def __init__(self,name, parent, right, left, label):
        self.name = name
        self.parent= parent
        self.right = right
        self.left = left
        self.label= label
        
def gini(n):
    s=sum(n)
    if s==0:
        return 0
    else:
        x=1.0
        for i in n:
            x-= (i/s)**2    
        return x

def count(df,feature):
    c1=df[df[feature]==1].shape[0]
    c0=df[df[feature]==0].shape[0]
    return c0,c1

def prob(c0,c1):
    p0=c0/(c0+c1)
    p1=c1/(c0+c1)
    return p0,p1

def deletFeature(df,feature):
    # delete selected feature from the data
    df_new=df.drop(feature,axis=1)
    return df_new


def gain(Ua,U0,U1,p0,p1):
    g=Ua-(p0*U0) - (p1*U1)
    return g

# find root node with remaining feature from data using gini and gain
def FindRoot(df):
    y='class' # target feature
    # list of features
    featureList=list(df.columns)
    # compute gini Ua
    c=count(df,y)
    Ua=gini(c)
    g=[] # to store all gini values for all features
# save in separate variable
    for i in featureList[0:-1]:
        # split the data into locally
        df0=df[df[i]==0]
        df1=df[df[i]==1]
        # compute gini for df0 
        c=count(df0,y)
        U0=gini(c)
        # compute gini for df1 
        c=count(df1,y)
        U1=gini(c)
        
        # probability
        c1=df1.shape[0]
        c0=df0.shape[0]
        p0=c0/(c0+c1)
        p1=c1/(c0+c1)
        g.append(gain(Ua,U0,U1,p0,p1))
        
    I=g.index(max(g))
    root=CreateNode(featureList[I],'None','None', 'None', 'None')
    return root


def split(node, max_depth, depth, df,l):
    # check for max depth
    if depth >= max_depth:
      
      node.left = CreateNode('class', node, 'None','None', 'None')  
      node.right = CreateNode('class', node, ''0,'None', 'None') 
      node.left.label = to_terminal(df)
      node.right.label = to_terminal(df)  
      l.append(node.left)
      l.append(node.right)
      return 
    else:

        # process left child
        #updates df before sending
        dfL=df[df[node.name]==0]
        dfL=dfL.drop([node.name], axis=1)
        node.left = FindRoot(dfL)
        l.append(node.left)
        split(node.left, max_depth, depth+1,dfL,l)
        # process right child
        #updates df before sending
        dfR=df[df[node.name]==1]
        dfR=dfR.drop([node.name], axis=1)
        node.right = FindRoot(dfR)
        l.append(node.right)
        split(node.right, max_depth, depth+1,dfR,l)
    
    return l
    
def to_terminal(df):
   c1=df[df['class']==1].shape[0]
   c0=df[df['class']==0].shape[0]
   p1=c1/(c0+c1)
   p0=c0/(c0+c1)
   if p1>p0:
       label='1'
       return label
   else:
       label='0'
       return label

def build_tree(train, max_depth):
   root = FindRoot(train)
   temp = [root]
   tree=split(root, max_depth, 0, train,[])
   tree=temp+tree
  
   return tree

def output(df):
    c1=df[df['class']==1].shape[0]
    c0=df[df['class']==0].shape[0]
    p1=c1/(c0+c1)
    p0=c0/(c0+c1)
    if p1>p0:
        label='1'
    else:
        label='0'
    
    return label

# prediction based on new example
#def predict(tree,newE,node):
#    print(node.name) 
#    if node.name!=None:
#        if newE[node.name]==0:
#            if tree[node.name][0].name=='class':
#                return tree[] 
#                
#            predict(tree,newE,node)
#        else:
#            node=tree[node.name][1]
#            predict(tree,newE,node)
#    
#    

if __name__ == '__main__':
    # read all data
    train=pd.read_csv('pa3_train.csv')
#   test=pd.read_csv('pa3_test.csv')
    val=pd.read_csv('pa3_val.csv')
    tree=build_tree(train,2)
    count=0
    for i in tree:
        count+=1
        print(count,i.name)
# list to dict
#    D={}
#    for i in tree:
#        if (i.left != 'None') or (i. right!='None'):
#            D[i.left.name+i.name]=[i.left,i.right,i.Llabel,i.Rlabel]
#            D[i.name+i.right.name]=[i.left,i.right,i.Llabel,i.Rlabel]

#    newE=val.loc[1]
#    y_hat=predict(D,newE,tree[0])
#    print(y_hat)
#    