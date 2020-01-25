from parser import Parser
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
import numpy as np
import time
from threading import Thread


errs=[]
class NetworkThread(Thread) :
  
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
    self.printError()

  def printGraph(self):
    self.nn.save_trts_err('./out/grid_search_graph/3_trts_err{}.png'.format(self._eta))
    self.nn.save_trts_acc('./out/grid_search_graph/3_trts_acc{}.png'.format(self._eta))
  
  def printError(self):      
    errs.append("eta [{}] momentum [{}] :{} \n".format(self._eta,self._momenutm,self.err))   
      




if __name__ == "__main__":  
    n_threads=10    
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
    _momentum=float(0.1)
    _eta=float(0.1)
    _lambda=float(0.01)
    threads=[]
    for i in range(1,n_threads):
        for i in range(1,n_threads):           
            for i in range(1,n_threads):
                t=NetworkThread(_eta,_momentum,topology,f,loss,dim_hid,tr,ts,vl,_lambda)
                t.start()
                threads.append(t)
                _lambda=round((_lambda+0.01),2)
            _momentum=round((_momentum+0.1),1)
        _eta=round((_eta+0.1),1)
    
    
    count=0
    for t in threads:    
      t.join()
      count+=1
      print("Count:{}".format(count))
   
    
    content=""
    for s in errs:
        content+=s+" \n"
    f = open('out/grid_search_error/error.txt', "a")
    f.write(content)     
    f.close()


        
    # for i in range(1,len(threads)):      
    #     threads[i-1].start()
    # for i in range(1,len(threads)):      
    #     threads[i-1].join()
    # for i in range(1,len(threads)):      
    #     threads[i-1].printError()    
    





