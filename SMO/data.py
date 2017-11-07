#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def GenerateData(nums, Distribution_Type = "Guassion"):
    DataCreated = []
    DataLabel = []
    for i in range(int(nums/2)):
        if "Liner" == Distribution_Type:
            temp = 10*np.random.random_sample((2, 1))
        if "Guassion" == Distribution_Type:
            temp = [10 * np.random.normal(0, 0.2, 1), 10 * np.random.normal(1, 0.2, 1)]
        temp = [temp[0][0], temp[1][0]]
        DataLabel.append(1)
        DataCreated.append(temp)

    for i in range(int(nums/2)):
        if "Liner" == Distribution_Type:
            temp = 10*np.random.random_sample((2, 1))
        if "Guassion" == Distribution_Type:
            temp = [10 * np.random.normal(1, 0.2, 1), 10 * np.random.normal(0, 0.2, 1)]
        temp = [temp[0][0], temp[1][0]]
        DataLabel.append(-1)
        DataCreated.append(temp)

    DataCreated = np.mat(DataCreated)
    DataLabel = np.mat(DataLabel).T
    #print DataCreated
    #print DataLabel
    #ShowData(DataCreated, DataLabel)
    return DataCreated, DataLabel

def ShowData(Data, Datalabel, IsShow = True):
    Data_rows = Data.shape[0]
    Datalabel_rows = Datalabel.shape[0]
    if Data_rows != Datalabel_rows:
        print("ShowData: Data_rows!=Datalabel_rows")
        return
    if IsShow:
        plt.figure("the input data")
        for i in range(Data_rows):
            if 1 == Datalabel[i]:
                plt.plot(Data[i][0], Data[i][1], 'r^')
            if -1 == Datalabel[i]:
                plt.plot(Data[i][0], Data[i][1], 'gs')
        #plt.ion()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.grid(True)
        plt.title("input data distribution")
        plt.show()
    else:
        print("Show Data has been denied")

def SaveData2Txt(Data, Datalabel, filepath = "Data.txt"):
    if "Data.txt" == filepath:
        print("there is no input file-path name, will generate in locally\n")
    try:
        myfile = open(filepath, 'w')
        rows = Data.shape[0]
        for i in range(rows):
            myfile.writelines(Data[i][0].__str__() + "\t" + Data[i][1].__str__() + "\t" +
                              Datalabel[i].__str__() + "\n")
        myfile.close()
    except IOError as e:
        print("write data to " + filepath + " failed!")

def ReadData2Mat(filepath = "Data.txt"):
    if "Data.txt" == filepath:
        print("will open %s as Data input\n" % (filepath))
    Data = []
    Label = []
    try:
        myfile = open(filepath, 'r')
        count = 0
        for line in myfile.readlines():
            lineArr = line.strip().split('\t')
            Data.append([float(lineArr[0]), float(lineArr[1])])
            Label.append([float(lineArr[2])])
            count += 1
        myfile.close()
    except IOError as e:
        print("open and read %s error\n" % (filepath))

    Data = np.mat(Data)
    Label = np.mat(Label)
    return Data, Label