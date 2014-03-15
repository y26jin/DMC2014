import sys
import psycopg2
import random
import pydot
from svmutil import *

#  QUESTION NEED TO CONSIDER
#  1. scale training/test data?
#  2. which svm classification to use?(C-support, one-class, etc)
#  3. add more features?

def main():
    y,x = svm_read_problem('./training_temp.dat')
#    y, x = svm_read_problem('../heart_scale')
    problem = svm_problem(y,x)
    
    print "TRAIN 1:"
    param1 = svm_parameter('-s 0 -b 1 -h 0') # C-support with prob
    m11 = svm_train(problem, param1)

    y_val, x_val = svm_read_problem('./validation_temp.dat')

    p_labs1, p_acc1, p_vals1 = svm_predict(y_val, x_val, m11, '-b 1')

    print "Test data result:"
    for l in p_labs1:
        print l

    print "Probability Estimate:"
    for ll in p_vals1:
        print ll

if __name__=="__main__":
    main()




