import sys

outputPath = "./MLZyNet/MLZyNet.sim/sim_1/behav/xsim/"
headerFilePath = "./MLZyNet/MLZyNet.sim/sim_1/behav/xsim/"

try:
    import cPickle as pickle
except:
    import pickle
import gzip
import numpy as np

dataWidth = 16                    #specify the number of bits in test data
IntSize = 1 #Number of bits of integer portion including sign bit

try:
    testDataNum = int(sys.argv[1])
except:
    testDataNum = 3

def DtoB(num,dataWidth,fracBits):                        #funtion for converting into two's complement format
    #print(num+1)
    if num >= 0:
        num = num * (2**fracBits)
        d = int(num)
    else:
        num = -num
        num = num * (2**fracBits)        #number of fractional bits
        num = int(num)
        if num == 0:
            d = 0
        else:
            d = 2**dataWidth - num
    return d
def separator(test_data3):
    test_data2=test_data3.split(",")
    
    #test_data4 = test_data.astype(float)

    return (test_data2)

def load_data():
    f = open('testDataexport_validate.csv', 'r')
    dataString = f.read()
    test_data3=dataString.split("\n")
    test_data=[]
    for i in range (0, len(test_data3), 1):
        x=separator(test_data3[i])
        test_data.append(x)
        

    #test_data = np.array([test_data2])
    #for i in range (0, len(test_data),1):
    
    f.close()
    return (test_data)

def genTestData(dataWidth,IntSize,testDataNum):
  
    dataHeaderFile = open(headerFilePath+"dataValues.h","w")
    dataHeaderFile.write("int dataValues[]={")
    te_d = load_data()

    test_inputs = [np.reshape(x, (1, 240)) for x in te_d]


    x = len(test_inputs[0][0])
    d=dataWidth-IntSize
    count = 0
    fileName = 'test_data.txt'
    f = open(outputPath+fileName,'w')
    fileName = 'visual_data'+str(te_d[1][testDataNum])+'.txt'
    g = open(outputPath+fileName,'w')
    k = open('testData.txt','w')
    for i in range(0,x):
        k.write(str(test_inputs[testDataNum][0][i])+',')
        test_inputs2=float(test_inputs[testDataNum][0][i])
        dInDec = DtoB(test_inputs2,dataWidth,d)
        myData = bin(dInDec)[2:]
        dataHeaderFile.write(str(dInDec)+',')
        f.write(myData+'\n')
        if test_inputs2>0:
            g.write(str(1)+' ')
        else:
            g.write(str(0)+' ')
        count += 1
        if count%28 == 0:
            g.write('\n')
    k.close()
    g.close()
    f.close()
    dataHeaderFile.write('0};\n')
    dataHeaderFile.write('int result='+str(te_d[1][testDataNum])+';\n')
    dataHeaderFile.close()
        
        
def genAllTestData(dataWidth,IntSize):
    te_d = load_data()
    #for i in range (0, 2, 1):
    test_inputs = [np.reshape(x, (1, 240)) for x in te_d]
    #test_inputs = test_inputs.astype(float)
    x = len(test_inputs[0][0])

    d=dataWidth-IntSize
    for i in range(len(test_inputs)):
        if i < 10:
            ext = "000"+str(i)
        elif i < 100:
            ext = "00"+str(i)
        elif i < 1000:
            ext = "0"+str(i)
        else:
            ext = str(i)
        fileName = 'test_data_'+ext+'.txt'
        f = open(outputPath+fileName,'w')
        for j in range(0,x):
            test_inputs2=float(test_inputs[i][0][j])
            dInDec = DtoB(test_inputs2,dataWidth,d)
            myData = bin(dInDec)[2:]
            f.write(myData+'\n')
        #f.write(bin(DtoB((te_d[1][i]),dataWidth,0))[2:])
        f.close()


if __name__ == "__main__":
    genTestData(dataWidth,IntSize,testDataNum=0)
    #genAllTestData(dataWidth,IntSize)
