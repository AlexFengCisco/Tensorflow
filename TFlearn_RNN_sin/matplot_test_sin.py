import numpy as np
import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

y_sin=np.sin(np.linspace(0,10,1000,dtype=np.float32))
print len(y_sin)
x = 0 
#for i in y_sin:
#    plt.plot(x,i,'bs',label='sin')
#    x=x+1
sin_plot=plt.plot(y_sin,label='sin')
plt.legend([sin_plot],['sin'])
#plt.show()
plt.savefig('sin.png')