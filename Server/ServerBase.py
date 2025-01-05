
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy

class Server(object):
    def __init__(self,args, global_model,Loaders_train, Loaders_local_test, Loader_global_test, logger, device):
        self.global_model = global_model
        self.args = args
        self.Loaders_train = Loaders_train
        self.Loaders_local_test = Loaders_local_test
        self.global_testloader = Loader_global_test
        self.logger = logger
        self.device = device
        self.LocalModels = []
        self.local_model_states={}

        self.lambdas ={}
        self.previous_losses ={}
        # 初始化每个客户端的lambda值和上一个损失
        # for idx in range(self.args.num_clients):
        #     self.lambdas[idx] = 1
        #     self.previous_losses[idx] = float('inf')



    def global_test_accuracy(self):
        self.global_model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.global_testloader):
            X = X.to(self.device)
            y = y.to(self.device)
            _,p = self.global_model(X)
            y_pred = p.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt
    
    
    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)
    
