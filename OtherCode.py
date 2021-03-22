import os
refPath  = '輸入參考路徑'
fileList = os.listdir(refPath)
tagPath = '輸入目標路徑'
for file in fileList:
    folder = os.path.exists(tagPath+file)
    if not folder:
        os.makedirs(tagPath+file)
        print('-----建立成功-----')
    else:
        print(file + '已存在')
