import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

path = "Datasets/"#path 지정
file_list = os.listdir(path)#path에 있는 모든 파일(디렉토리) 리스트를 가져온다.
file_list_txt = [file for file in file_list if file.endswith(".txt")]#.txt 데이터만 가져온다.
print("file_list: {}".format(file_list_txt))
def load_data(txt_data):
    data = np.loadtxt("Datasets/{}".format(txt_data))
    print(txt_data)
    print("type of data: ", type(data))
    print("shape of data: ", data.shape)
    print("data: ", data)
    data = pd.DataFrame(data=data)
    return data
i=0
# for i in file_list_txt:
#     load_data(i)
while(True):
    data = load_data(file_list_txt[i])

    fig = plt.figure(figsize = (20, 10))

    #판다스 객체를 plot 함수에 입력
    plt.plot(data)

    #차트 제목 추가
    plt.title("{}".format(file_list_txt[i]))
    #축 이름 추가
    plt.xlabel('time')
    plt.ylabel('value')

    plt.show()

    i+=1
    if(i==len(file_list_txt)):
        break

