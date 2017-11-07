import numpy as np

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        #用来缓存误差
        #是两列的，第一列表示是否有效，第二列是实际的E值

def calcEk(oS,k):
    #用来计算误差的函数
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    #在确定好第一个alpha的情况下，确定第二个
    #求找最大步长，E1-E2
    maxK=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    #将Ei设置为有效
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    #nonzero返回一个列表，这个列表中包含以输入列表为目录的列标识
    #返回非零E值所对应的alpha值
    #因为在eCache的第一列代表是否有效，非0代表有效
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        #如果都不满足要求，直接随机选一个
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
     Ek=calcEk(oS,k)
     oS.eCache[k]=[1,Ek]

def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol)and (oS.alphas[i]<oS.C))or((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        #启发式方法选择第二个alpha
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H :
            print('L==H')
            return 0
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        #计算eta
        if eta>=0:
            print('eta>=0')
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        #计算未经剪辑的最优解
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        #增加约束条件
        updateEk(oS,j)
        #更新对应的误差值
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print('j is not moving enough')
            return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        #由第二个alpha求解另一个alpha
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T\
                   -oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T\
                   -oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    #初始化数据结构
    iter=0
    entireSet=True
    alphaPairChanged=0
    while(iter<maxIter) and ((alphaPairChanged>0)or(entireSet)):
        #循环条件：1、迭代次数少于最大迭代数；2、遍历着数据集对alpha进行了改变
        alphaPairChanged=0
        if entireSet:
            for i in range(oS.m):
                #oS.m表示数据的个数
                alphaPairChanged+=innerL(i,oS)
                #此处i对数据集进行遍历，InnerL选择第二个alpha，如果有改变返回1，否则返回0
            print('fullSet,iter: %d i: %d, Pairs changed: %d' %(iter,i,alphaPairChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            #找所有非零的非边界值
            for i in nonBoundIs:
                alphaPairChanged+=innerL(i,oS)
                print('non-bound,iter: %d i: %d, Pairs changed: %d' %(iter,i,alphaPairChanged))
            iter+=1
        if entireSet:entireSet=False
        elif (alphaPairChanged==0):entireSet=True
        print('iteration number: %d' %iter)
    return oS.b,oS.alphas

def selectJrand(i,m):
    #i是第一个alpha的下标，m都是alpha的总个数
    j=i
    while(j==i):
        #j是第二个alpha的下标
        #如果j与i相同，则重新选取一个
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    #设置上下界限
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj