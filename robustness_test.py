import os

# run simulations, remember to turn on the artifacts for robustness test
for i in range(20):
    os.system('python main.py --output data/outout_{:d}.avi --artifact --plot'.format(i))