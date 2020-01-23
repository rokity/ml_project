from parser import Parser
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
import numpy as np
import time
from threading import Thread
import os


class NetworkThread (Thread):
  
  def __init__(self,eta,momentum,topology,f_act,loss,dim_hid,tr,ts,vl,lam):
    Thread.__init__(self)
    self._eta = eta
    self._momenutm = momentum
    self._topology=topology
    self._f_act=f_act
    self._loss=loss
    self._dim_hid=dim_hid
    self._tr=tr
    self._ts=ts
    self._vl=vl
    self._lambda=lam
    
  def run(self):
    self.nn = NeuralNetwork(self._topology, self._f_act, self._loss, self._dim_hid, self._tr.size, self._eta, self._momenutm,self._lambda)
    self.err = self.nn.train(self._tr,self._tr, self._ts, 0.02, 2000)    
    print("fatto")

  def printGraph(self):
    self.nn.save_trts_err('./out/grid_search_graph/3_trts_err{}.png'.format(self._eta))
    self.nn.save_trts_acc('./out/grid_search_graph/3_trts_acc{}.png'.format(self._eta))
  
  def printError(self):
      f = open('out/grid_search_error/error.txt', "a")
      f.write("{}:{} \n".format(self._eta,self.err))     
      f.close()

def file_error_init():
    path='out/grid_search_error/error.txt'
    if (os.path.exists(path)):
        os.remove(path)
        with open(path, 'w'): pass
    else:
        with open(path, 'w'): pass


def main():    
    n_threads=3
    file_error_init()
    path_tr = 'monks/monks-3.train'
    path_ts = 'monks/monks-3.test'
    dim_in = 6
    one_hot = 17
    dim_hid = 4
    dim_out = 1
    f = FunctionsFactory.build('sigmoid')
    loss = FunctionsFactory.build('lms')
    if one_hot is None:
        topology = [dim_in, dim_hid, dim_out]
    else:
        topology = [one_hot, dim_hid, dim_out]
    tr, vl, ts = Parser.parse(path_tr, path_ts, dim_in, dim_out, one_hot, None)
    threads=[]
    _eta=float(0.1)
    for i in range(1,n_threads):
        threads.append(NetworkThread(_eta,0.5,topology,f,loss,dim_hid,tr,ts,vl,0.01))
        _eta=round((_eta+0.1),1)  
    for i in range(1,n_threads):      
        threads[i-1].start()
    for i in range(1,n_threads):      
        threads[i-1].join()
    for i in range(1,n_threads):      
        threads[i-1].printGraph()
    for i in range(1,n_threads):      
        threads[i-1].printError()    
    


if __name__ == "__main__":
    main()

