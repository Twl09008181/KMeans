import numpy as np
import numpyToCPP as ntcpp



def testNonCopy():
    a = np.array([[1,2],[3,4]],dtype=np.float64)
    ntcpp.square(a)

    assert(a[0][0] == 1)
    assert(a[0][1] == 4)
    assert(a[1][0] == 9)
    assert(a[1][1] == 16)
