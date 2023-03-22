from zynet import zynet
from zynet import utils
import numpy as np

def genMnistZynet(dataWidth,sigmoidSize,weightIntSize,inputIntSize):
    model = zynet.model()
    model.add(zynet.layer("flatten",240))
    model.add(zynet.layer("Dense",20,"relu"))
    model.add(zynet.layer("Dense",20,"relu"))
    weightArray = utils.genWeightArray('WeightsAndBiases_Ahmed.txt')
    biasArray = utils.genBiasArray('WeightsAndBiases_Ahmed.txt')
    model.compile(pretrained='Yes',weights=weightArray,biases=biasArray,dataWidth=dataWidth,weightIntSize=weightIntSize,inputIntSize=inputIntSize,sigmoidSize=sigmoidSize)
   # zynet.makeXilinxProject('MLZyNet','xczu49dr-ffvf1760-2-e')
   # zynet.makeIP('MLZyNet')
   # zynet.makeSystem('MLZyNet','myBlock2')
    
if __name__ == "__main__":
    genMnistZynet(dataWidth=16,sigmoidSize=10,weightIntSize=4,inputIntSize=1)