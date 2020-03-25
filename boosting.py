import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        e = 0
        for y in np.unique(Y):
            add = np.sum(D[Y == y])
            # print(np.sum(D))
            if add != 0:
                e -= add/np.sum(D) * math.log(add/np.sum(D), 2)

        #########################################
        return e 
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        cntx = dict()
        cntd = dict()
        ce = 0
        res = zip(X, Y, D)

        for x, y, d in res:
            if x not in cntx:
                cntx[x] = [y]
                cntd[x] = [d]
            else:
                cntx[x].append(y)
                cntd[x].append(d)
        for a, b in cntx.items():
            ce += np.sum(D[X == a]) * DS.entropy(np.array(b), np.array(cntd[a]))


        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        g = DS.entropy(Y, D) - DS.conditional_entropy(Y, X, D)

        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        cp = DT.cutting_points(X, Y)
        th = g = -1

        if type(cp) == float:
            return -float('Inf'), -1

        for c in cp:
            helper = []
            for x in X:
                if x > c:
                    helper.append('L')
                else:
                    helper.append('S')
            # print(DS.entropy(Y, D), DS.conditional_entropy(Y, helper, D))
            # print(helper)
            helper = np.asarray(helper)
            gg = DS.information_gain(Y, helper, D)

            if gg > g:
                th = c
                g = gg
        print(th, g)



        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g_all = []
        th_all = []
        for idx in range(len(X)):
            th, g = DS.best_threshold(X[idx], Y, D)
            g_all.append(g)
            th_all.append(th)
        i = g_all.index(max(g_all))
        th = th_all[i]
        #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        list = dict()
        for e in Y:
            if e not in list:
                list[e] = sum(D[Y==e])

        for y, time in list.items():
            if time == max(list.values()):
                return y
        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y)
        t.p = DS.most_common(Y, D)
        # if Condition 1 or 2 holds, stop splitting
        if DT.stop1(Y) or DT.stop2(X):
            t.isleaf = True
            return t

        # find the best attribute to split
        t.i, t.th = self.best_attribute(X, Y, D)

        # configure each child node
        t.C1, t.C2 = self.split(t.X, t.Y, t.i, t.th)

        D1,D2 = [],[]
        for j in range(len(D)):
            if X[t.i, j] < t.th:
                D1.append(D[j])
            else:
                D2.append(D[j])
        D1 = np.array(D1)
        D2 = np.array(D2)



        t.C1.p = DS.most_common(t.C1.Y, D1)
        t.C2.p = DS.most_common(t.C2.Y, D2)

        t.C1.isleaf = True
        t.C2.isleaf = True


        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        for i in range(len(Y)):
            if Y[i] != Y_[i]:
                e += D[i]
        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        try:
            a = 0.5 * math.log((1 - e) / e)
        except ZeroDivisionError: # e = 0
            a = 101
        except ValueError:  # e = 1
            a = -101
        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        DD = np.zeros(len(D))
        for i in range(len(Y)):
            if Y[i] != Y_[i]:
                DD[i] = D[i] * np.exp(a)
            else:
                DD[i] = D[i] * np.exp(-a)
        D = DD/sum(DD)
        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ds = DS()
        t = ds.build_tree(X, Y, D)
        Y_ = np.array([])
        for i in range(len(X[0])):
            attr=X[:, i]
            Y_ = np.append(Y_, DS.inference(t, attr))
        e = AB.weighted_error_rate(Y, Y_, D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D, a, Y, Y_)
        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given an adaboost ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        lab = np.array([])
        for t in T:
            lab = np.append(lab, DT.inference(t, x))

        stat = []
        for i in range(len(lab)):
            stat.append([lab[i],A[i]])

        d = dict()
        for i in range(len(lab)):
            if lab[i] not in d:
                d[lab[i]] = A[i]
            else:
                d[lab[i]] += A[i]

        y = DT.most_common(d)
        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for x in X.T:
            Y.append(AB.inference(x, T, A))
        Y = np.array(Y)

        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE



        # initialize weight as 1/n
        D = np.ones(len(X[0])) / len(X[0])
        # iteratively build decision stumps
        T = []
        A = []
        for i in range(n_tree):
            t, a, D = AB.step(X, Y, D)
            T.append(t)
            A.append(a)

        T = np.array(T)
        A = np.array(A)

        #########################################
        return T, A
   



 
