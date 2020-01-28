from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *
import numpy as np
import time
import multiprocessing
import datetime

results=[]
def run(nn,tr,vl,ts):
  err = nn.train(tr, vl, ts, 1e-2, 1000)
  results.append([err,[nn.get_eta(),nn.get_momentum(),nn.get_lambda()]])

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

    acc = FunctionsFactory.build('accuracy')
    if one_hot is None:
      topology = [dim_in, dim_hid, dim_out]
    else:
      topology = [one_hot, dim_hid, dim_out]
    parser = Monks_parser(path_tr, path_ts)
    tr, vl, ts = parser.parse(dim_in, dim_out, one_hot, 0.3)
    _momentum=float(0.1)
    _eta=float(0.1)
    _lambda=float(0.01)
    threads=[]
    count=0
    for i in range(1,n_threads):
        for i in range(1,n_threads):           
            for i in range(1,n_threads):
                nn = NeuralNetwork(topology, f, loss, acc, dim_hid, tr.size, _eta, _momentum, _lambda)
                threads.append(nn)
                              
                _lambda=round((_lambda+0.01),2)
            _momentum=round((_momentum+0.1),1)
            _lambda=float(0.01)
        _eta=round((_eta+0.1),1)
        _momentum=float(0.1)
        _lambda=float(0.01)
    process=[]
    while(count<len(threads)):
      for i in range(0,10):
        nn=threads[count]
        p = multiprocessing.Process(target=run, args=(nn,tr,vl,ts))
        p.start()
        print("lancio {}".format(count))
        process.append(p)
        count+=1
      print(datetime.datetime.now())      
      while(len(process)>0):
        p=process.pop()
        p.join()
        p.close()
        print("finito {}".format(count))
      print(datetime.datetime.now())

    

    
    print("finito")
    
   
    
    content=""
    err_min=[1.0,None]
    for s in results:
        if(s[0]<err_min[0]):
          err_min[0]=s[0]
          err_min[1]=s[1]
        content+="{} Parametri : Eta -> {} Momentum -> {} Lambda-Regolarizzatore -> {} \n".format(s[0],s[1][0],s[1][1],s[1][2])
    content+="{} Valore Perfetto : Eta -> {} Momentum -> {} Lambda-Regolarizzatore -> {} \n".format(err_min[0],err_min[1][0],err_min[1][1],err_min[1][2])
    f = open('out/grid_search_error/error.txt', "a")
    f.write(content)         
    f.close()

    





