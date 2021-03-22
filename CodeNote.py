import os

os.listdir("filepath")
os.path.exists("filepath")
os.makedirs("filepath")

import shutil

shutil.move("oldfile","newfile")
shutil.copyfile("oldfile","newfile") # 只能是文件
shutil.copy("oldfile","newfile")  # # oldfile只能是文件夹，newfile可以是文件，也可以是目錄


# String process
list = 'I love you'
list.split(' ',1) 
list.rsplit(' ')
