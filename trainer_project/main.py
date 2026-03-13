import numpy
print(numpy.__version__)
   # 应该输出 1.24.4
import numpy.core._multiarray_umath  # 不应再报错
