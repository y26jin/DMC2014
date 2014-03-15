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
#    print "C = ",sys.argv[1], "epi = ", sys.argv[2]

    # convert system input argument to floating point
 #   temp = float(sys.argv[1]) + float(sys.argv[2]) 

    y,x = svm_read_problem('./training.dat')
    problem = svm_problem(y,x)
    
    print "TRAIN 1:"
    param1 = svm_parameter('-s 0 -b 1 -h 0') # C-support with prob
    m11 = svm_train(problem, param1)
    param1_cross = "-v 3"
    m12 = svm_train(problem, param1_cross)

    #print "TRAIN 2:"
    #param2 = svm_parameter('-s 3 -c 0.25 -h 0')
    #m21 = svm_train(problem, param2)
    #param2_cross = "-v 3"
    #m22 = svm_train(y, x, param2_cross)

    #print "TRAIN 3:"
    #param3 = svm_parameter('-s 3 -c 0.5 -h 0')
    #m31 = svm_train(problem, param3)
    #param3_cross = "-v 3"
    #m32 = svm_train(y, x, param3_cross)


#    CV_ACC = svm_train(y,x, '-v 3')

    y_val, x_val = svm_read_problem('./validation.dat')

    p_labs1, p_acc1, p_vals1 = svm_predict(y_val,x_val,m11,'-b 1')
    print "Test data result:"
    print p_labs1
    print "Probability Estimate:"
    print p_vals1
#    print y
 #   print "===="
  #  print p_labs1

#    p_labs2, p_acc2, p_vals2 = svm_predict(y,x,m21)
 #   p_labs3, p_acc3, p_vals3 = svm_predict(y,x,m31)
    

if __name__=="__main__":
    main()




