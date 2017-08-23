import main
import optimizer

optimizer_name = "AdamOptimizer"
optimizer_imp = main.get_implementation(optimizer.Optimizer, optimizer_name)
print (optimizer_imp)