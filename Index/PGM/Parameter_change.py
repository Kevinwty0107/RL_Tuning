import os
from sre_constants import SUCCESS
import numpy as np
import torch
import gym
import argparse
import os

def updateFile(file,epsilon,ER):
    file_data = ""
    flags = os.path.exists("./parameters.txt")
    if flags:
        [old_epsilon,old_ER] = np.loadtxt("./parameters.txt")
        if [old_epsilon,old_ER] == []:
            print("Fail to replace parameters")
    else:
        old_epsilon,old_ER=[30,9]


    with open(file, "r") as f:
        for line in f:
            line = line.replace("pgm::PGMIndex<K,%d, %d, float> pgm;"%(old_epsilon,old_ER),"pgm::PGMIndex<K,%d, %d, float> pgm;" %(epsilon,ER) )
            file_data += line
    with open(file,"w") as f:
        f.write(file_data)

    np.savetxt("./parameters.txt",[epsilon,ER],fmt='%d')

if __name__ == "__main__":

    # os.chdir("./")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=int, default=50, help="Epsilon for PGM")
    parser.add_argument("--ER", default=100, type=int)             
    args = parser.parse_args()
    updateFile("./Index/PGM/index_test.cpp",args.epsilon,args.ER)