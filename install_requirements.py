import os
f = open('requirements_NEW.txt', 'r')
for line in f.readlines():
    print(line)
    os.system('conda install -c conda-forge ' + line)