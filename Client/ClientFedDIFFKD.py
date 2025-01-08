import   进口 numpy as np
import   进口 torch
import   进口 scipy
from   从 torch.utils.data import   进口 Dataset
import   进口 torch
import   进口 copy
import   进口 torch.nn as nn
from   从 sklearn.cluster import   进口 KMeans
import   进口 torch.optim as optim
import   进口 torch.nn.functional as f
from   从 utils import   进口 Accuracy, soft_predict
from   从 Client.ClientBase import   进口 Client
import   进口 gc
 

class   类 ClientFedDIFFKD(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """

    def __init__(self, args, model, Loader_train, loader_test, idx, logger, code_length, num_classes, device):
        super().__init__(args, model, Loader_train, loader_test, idx, logger, code_length, num_classes, device)

    def update_weights(self, global_round):Def update_weights(self, global_round)：
        self.model.to   来(self.device)
        self.model.train   火车()
        epoch_loss = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        for   为 iter in   在 range(self.args.local_ep):
            batch_loss = []
            for   为 batch_idx, (X, y) in   在 enumerate(self.trainloader):
                X = X.to   来(self.device)
                y = y.to   来(self.device)
                optimizer.zero_grad()
                _, p = self.model(X)
                loss = self.ce(p, y)
                loss.backward()
                if   如果 self.args.clip_grad != None   没有一个:
                    nn.utils   跑龙套.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                optimizer.step()
                if   如果 batch_idx % 10 == 0:
                    print(
                        '| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, self.idx, iter, batch_idx * len(X),
                            len(self.trainloader.dataset   数据集),
                                                          100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger   日志记录器.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_DIFFKD(self, global_features, global_soft_prediction, lam, gamma, lam3, temp, global_round)：def update_weights_DIFFKD(self, global_features, global_soft_prediction, lam, gamma, lam3, temp, global_round):
        self.model.to   来(self.device)
        self.model.train   火车()
        self.teacher_model.to   来(self.device)
        self.teacher_model.eval()
        epoch_loss = []
        epoch_loss3 = []
        epoch_loss4 = []
        epoch_G = []
        epoch_loss2 = []
        epoch_loss5 = []
        epoch_G1 = []
        # optimizer = optim.Adam(list(self.model.parameters()) + list(self.diffkd_feat.parameters()), lr=self.args.lr)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        optimizer1 = optim.Adam(self.diffkd_feat.parameters(), lr=self.args.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.5)
        # tensor_global_features = self.dict_to_tensor(global_features).to(self.device)
        # tensor_global_soft_prediction = self.dict_to_tensor(global_soft_prediction).to(self.device)
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss2 = []
            batch_loss3 = []
            batch_loss4 = []
            batch_G = []

            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                optimizer1.zero_grad()
                F, Z = self.model(X)
                _,teacher_Z=self.teacher_model(X)
                # pooled_features = self.model.avgpool(tensor_global_features)
                # flattened_features = pooled_features.view(pooled_features.size(0), -1)
                # global_logits=self.model.classifier(flattened_features)
                loss1 = self.ce(Z, y)
                target_features = copy.deepcopy(F.data)

                for i in range(y.shape[0]):
                    if int(y[i]) in global_features.keys():
                        target_features[i] = global_features[int(y[i])][0].data
                # target_features = target_features.to(self.device)

                # student_feat = F.to(self.device)
                # teacher_feat = target_features
                student_feat_refined, ddim_loss_feat, G = \
                    self.diffkd_feat(F.to(self.device), target_features.to(self.device))
                loss3 = ddim_loss_feat
                loss4 = self.mse(student_feat_refined, target_features.to(self.device))


                loss2=f.kl_div(f.log_softmax(Z/temp,dim=1),f.softmax(teacher_Z/temp,dim=1),reduction='batchmean')
                loss = loss1  +lam*loss2+gamma*loss3+ lam3 * loss4

                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                    nn.utils.clip_grad_norm_(self.diffkd_feat.parameters(), max_norm=self.args.clip_grad)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.1)
                optimizer.step()
                optimizer1.step()
                if batch_idx % 10 == 0:
                    print(
                        '| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.6f} Loss2: {:.6f} Loss3: {:.6f}  Loss4: {:.6f}'.format(
                            global_round, iter, batch_idx,
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss1.item(),loss2.item(), loss3.item(),loss4.item()
                        ))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                batch_loss2.append(loss2.item())
                batch_loss3.append(loss3.item())
                batch_loss4.append(loss4.item())
                batch_G.append(G.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            epoch_loss3.append(sum(batch_loss3) / len(batch_loss3))
            epoch_loss4.append(sum(batch_loss4) / len(batch_loss4))
            epoch_G.append(sum(batch_G) / len(batch_G))

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss),sum(epoch_loss2) / len(epoch_loss2),sum(epoch_loss3) / len(epoch_loss3), sum(
            epoch_loss4) / len(epoch_loss4),sum(epoch_G) / len(epoch_G),
    # generate knowledge for FedDFKD
    def generate_knowledge(self, temp, global_features, global_round, global_soft_prediction):
        self.model.to(self.device)
        self.model.eval()
        local_features = {}
        local_soft_prediction = {}
        num_classes = self.model.num_classes
        features_avg = [torch.zeros(self.code_length).to(self.device)] * num_classes
        # soft_predictions_avg = [torch.zeros(num_classes).to(self.device)] * num_classes
        count = [0] * num_classes
        for batch_idx, (X, y) in enumerate(self.trainloader):
            X = X.to(self.device)
            y = y
            F, Z = self.model(X)
            # Q = soft_predict(Z, temp).to(self.device)
            m = y.shape[0]
            for i in range(len(y)):
                if y[i].item() in local_features:
                    local_features[y[i].item()].append(F[i, :])
                    # local_soft_prediction[y[i].item()].append(Z[i, :])
                else:
                    local_features[y[i].item()] = [F[i, :]]
                    # local_soft_prediction[y[i].item()] = [Z[i, :]]

            del X
            del y
            del F
            del Z
            # del Q
            gc.collect()

        # features_avg, soft_predictions_avg = self.local_knowledge_aggregation(local_features, local_soft_prediction,
        #                                                                       std=self.args.std)
        features_avg = self.local_knowledge_aggregation(local_features,std=self.args.std)
        if global_round >= 1:
            label_rations = {label: len(features) / len(self.trainloader.dataset) for label, features in
                             local_features.items()}
            # print(label_rations)
            for label, features in features_avg.items():
                # if label_rations[label] >= 0.1:
                global_features[label].append(features[0])
            #
            # for label, soft_predictions in soft_predictions_avg.items():
            #     # if label_rations[label] >= 0.1:
            #     global_soft_prediction[label].append(soft_predictions[0])

        else:
            for label, features in features_avg.items():

                if label in global_features:
                    global_features[label].append(features[0])
                else:
                    global_features[label] = [features[0]]
            # for label, soft_predictions in soft_predictions_avg.items():
            #     if label in global_soft_prediction:
            #         global_soft_prediction[label].append(soft_predictions[0])
            #     else:
            #         global_soft_prediction[label] = [soft_predictions[0]]

        # return (features_avg)

    def local_knowledge_aggregation(self, local_features, std):
        agg_local_features = dict()
        # agg_local_soft_prediction = dict()
        # feature_noise = std * torch.randn(self.args.code_len).to(self.device)
        for [label, features] in local_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                agg_local_features[label] = [feature / len(features)]
            else:
                agg_local_features[label] = [features[0].data]

        # for [label, soft_prediction] in local_soft_prediction.items():
        #     if len(soft_prediction) > 1:
        #         soft = 0 * soft_prediction[0].data
        #         for i in soft_prediction:
        #             soft += i.data
        #
        #         agg_local_soft_prediction[label] = [soft / len(soft_prediction)]
        #     else:
        #         agg_local_soft_prediction[label] = [soft_prediction[0].data]

        return agg_local_features

    def dict_to_tensor(self, dic):
        lit = []
        for key, tensor in dic.items():
            lit.append(tensor[0])
        lit = torch.stack(lit)
        return lit
