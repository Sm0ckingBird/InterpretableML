# SGD
# W += -LR * dx

# Momentum
# m = b1 * m - LR * dx  W += m

# AdaGrad
# v += dx^2   W += -LR * dx / (v^1/2)

# RMSProp
# v = b1 * v + (1-b1)*dx^2
# W += -LR * dx / (v^1/2)

# Adam
# m = b1 * m + (1-b1)*dx
# v = b2 * v + (1-b2)*dx^2
# W += -LR * m / (v^1/2)
