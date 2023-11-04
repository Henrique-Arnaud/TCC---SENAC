import shutil
import os

targetPath = r'./baseOficial/600img/colorida/50x50/'
destinationPath = r'./baseOficial/600img/colorida/50x50/fechado/'

mylist = os.listdir(targetPath)
mylist.sort()
count = 0
for index, file_name in enumerate(mylist):
  source = targetPath + file_name
  destination = destinationPath + 'fechado' + str(index) + '.jpg'
    # copy only files
  if os.path.isfile(source):
    shutil.move(source, destination)
    print(index, destination)
    count = count+1
print(count)