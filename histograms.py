from matplotlib.pyplot import axis, plot, show, bar, hist, grid, legend, yticks

train = ['Train'] * 14133
test = ['Test'] * 4362
data = train + test

bin_edges = [k-0.5 for k in range(3)]
hist(data, bin_edges, density=True, rwidth=0.9, color='green', edgecolor='black', alpha=0.5, label='Procent date')
distribution = dict([('Train', 0.8), ('Test', 0.2)])
bar(distribution.keys(), distribution.values(), width=0.85, color='red', edgecolor='black', alpha=0.6,
    label='Procent ideal')
yticks([i/1000 for i in range(0, 850, 50)])
legend(loc='upper right')
grid()
show()

fresh_apples = ['Fresh\nApples'] * (2424 + 791)
fresh_oranges = ['Fresh\nOranges'] * (1466 + 388)
fresh_banana = ['Fresh\nBanana'] * (2468 + 892)
rotten_apples = ['Rotten\nApples'] * (3248 + 988)
rotten_oranges = ['Rotten\nOranges'] * (1595 + 403)
rotten_banana= ['Rotten\nBanana'] * (2932 + 900)

data2 = fresh_apples + fresh_banana + fresh_oranges + rotten_apples + rotten_banana + rotten_oranges

bin_edges = [k-0.5 for k in range(7)]
hist(data2, bin_edges, density=True, rwidth=0.9, color='green', edgecolor='black', alpha=0.5, label='Procent fructe')
legend(loc='upper right')
yticks([i/10000 for i in range(0, 2500, 125)])
grid()
show()
