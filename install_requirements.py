import os
f = open('requirements.txt', 'r')
for line in f.readlines():
    print(line)
    os.system('conda install -c conda-forge ' + line)