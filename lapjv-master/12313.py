# from lap import lapjv
# import numpy as np
#
# a = np.array([[0.1, 0.6, 0.3], [0.2, 0.1, 0.6], [0.5, 0.2, 0.9]])
#
# x, y, c = lapjv(a)
#
# print(x, y, c)

from lap import lapjv
import numpy as np

cost_matrix = np.array([
    [1, 5, 3],
    [2, 3, 7],
    [5, 2, 9]
])

x_assign, y_assign, _ = lapjv(cost_matrix)
# 总代价最小
cost_value = 0
for _x, _y in zip([x_assign], [y_assign]):
    print("选择", _x, _y)
    cost_value += int(cost_matrix[_x, _y])
print("最小代价为：{}".format(cost_value))
