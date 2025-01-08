import   进口 wandb
from   从 torch.utils.data 从torch.utils.data导入数据集import Dataset
import   进口 torch   进口火炬
import   进口 copy   进口复制
from   从 utils import   进口 Accuracy
from   从 Server.ServerBase import   进口 Server
from   从 Client.ClientFedDIFFKD import   进口 ClientFedDIFFKD
from   从 tqdm import   进口 tqdm
import   进口 numpy    导入numpy为npas np
from   从 utils import   进口 average_weights
from   从 mem_utils import   进口 MemReporter
import   进口 time
from   从 sampling import   进口 LocalDataset, LocalDataloaders, partition_data
import   进口 gc   进口gc
 

class   类 ServerFedDIFFKD(Server):
    def __init__(self, args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device):
        super   超级().__init__(args, global_model, Loader_train, Loaders_local_test, Loader_global_test, logger, device)


    def Create_Clints(self):
        for   为 idx in   在 range   范围(self.args   arg游戏.num_clients):
            self.LocalModels.append   附加(
                ClientFedDIFFKD(self.args   arg游戏, copy.deepcopy(self.global_model), self.Loaders_train[idx],
                                self.Loaders_local_test[idx], idx=idx, logger=self.logger   日志记录器,
                                code_length=self.args   arg游戏.code_len, num_classes=self.args   arg游戏.num_classes,
                                device=self.device   设备))

    def add_to_global(self, global_dict, local_dict, total_samples, local_features_items, v=0.1):

        label_rations = {label: len(features) / total_samples for   为 label, features in   在 local_features_items}

        for   为 label, features_or_predictons in   在 local_dict.items():

            if   如果 label_rations[label] > v:

                if label not in global_dict:
                    global_dict[label] = features_or_predictons
                else:
                    global_dict[label].extend(features_or_predictons)
        return global_dict

    def global_knowledge_aggregation(self, global_features, global_soft_prediction):
        agg_global_local_features = dict()
        agg_global_local_soft_prediction = dict()
        #
        # for label,features in global_features.items():
        #     previous_aggregated_feature=features[0].data
        #     if len(features)>1:
        #         new_features_sum= sum(f.data for f in features[1:])
        #         new_features_avg=new_features_sum/(len(features)-1)
        #         updated_feature =(previous_aggregated_feature+new_features_avg)/2
        #     else:
        #         updated_feature=previous_aggregated_feature
        #     agg_global_local_features[label]=[updated_feature]
        #
        #
        # for label,soft_prediction in  global_soft_prediction.items():
        #     previous_aggregated_soft_prediction=soft_prediction[0].data
        #     if len(soft_prediction)>1:
        #         new_soft_prediction_sum= sum(f.data for f in soft_prediction[1:])
        #         new_soft_prediction_avg=new_soft_prediction_sum/(len(soft_prediction)-1)
        #         updated_soft_prediction =(previous_aggregated_soft_prediction+new_soft_prediction_avg)/2
        #     else:
        #         updated_soft_prediction=previous_aggregated_soft_prediction
        #     agg_global_local_soft_prediction[label]=[updated_soft_prediction]

        for [label, features] in global_features.items():
            if len(features) > 1:
                feature = 0 * features[0].data
                for i in features:
                    feature += i.data
                agg_global_local_features[label] = [feature / len(features)]
            else:
                agg_global_local_features[label] = [features[0].data]

        for [label, soft_prediction] in global_soft_prediction.items():
            if len(soft_prediction) > 1:
                soft = 0 * soft_prediction[0].data
                for i in soft_prediction:
                    soft += i.data

                agg_global_local_soft_prediction[label] = [soft / len(soft_prediction)]
            else:
                agg_global_local_soft_prediction[label] = [soft_prediction[0].data]

        return agg_global_local_features, agg_global_local_soft_prediction

    def adaptive_weigth_adjustment(self,idx,epoch,global_epoch):

        # previous_loss = self.previous_losses[idx]  # 直接获取，不再提供默认值
        lambda_value = self.lambdas[idx]  # 直接获取，不再提供默认值
        # if current_loss < previous_loss:
        #     #如果损失减少，减小教师模型的影响
        #     new_lambda=max(0.0,lambda_value-0.1)
        # else:
        #     #如果损失增加，增加教师模型的影响
        #     new_lambda=min(1.5,lambda_value+0.1)
        new_lambda=lambda_value+(epoch/global_epoch)*1
        self.lambdas[idx]= new_lambda
        # self.previous_losses[idx]=current_loss



    def train(self):
        global_features = {}
        global_soft_prediction = {}
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()

        for epoch in tqdm(range(self.args.num_epochs)):
            all_local_features = {}
            all_local_soft_prediction = {}
            Knowledges = []
            test_accuracy = 0
            local_weights, local_losses = [], []
            epoch_loss2 = []
            epoch_loss3 = []
            epoch_loss4 = []
            epoch_G = []


            print(f'\n | Global Training Round : {epoch + 1} |\n')
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
            for idx in idxs_users:
                # self.lambdas = {client.id: 0.5 for client in idxs_users}
                if self.args.upload_model == True:
                    self.LocalModels[idx].load_model(global_weights)
                if epoch < 1:
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc
                    self.LocalModels[idx].generate_knowledge(
                        temp=self.args.temp,
                        global_features=global_features
                        , global_round=epoch
                        , global_soft_prediction=global_soft_prediction)
                    # global_features.update(local_features)
                    # global_soft_prediction.update(local_soft_predictions)

                else:
                    #从保存的状态中获取上一轮的模型状态
                    last_model_state=self.local_model_states.get(idx)
                    #将上一轮的模型状态加载到客户端模型
                    self.LocalModels[idx].load_teacher_model(last_model_state)
                    # print(self.lambdas[idx])
                    w, loss,loss2, loss3, loss4,G = self.LocalModels[idx].update_weights_DIFFKD(global_round=epoch,
                                                                                           global_features=global_features,
                                                                                           global_soft_prediction=global_soft_prediction,
                                                                                           lam=self.args.lam,
                                                                                           gamma=self.args.gamma,
                                                                                           lam3=self.args.lam3,
                                                                                           temp=self.args.temp)

                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    epoch_loss2.append(copy.deepcopy(loss2))
                    epoch_loss3.append(copy.deepcopy(loss3))
                    epoch_loss4.append(copy.deepcopy(loss4))
                    epoch_G.append(copy.deepcopy(G))

                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc

                    self.LocalModels[idx].generate_knowledge(
                        temp=self.args.temp, global_features=global_features
                        , global_round=epoch
                        , global_soft_prediction=global_soft_prediction)

                # global_features.update(local_features)
                # global_soft_prediction.update(local_soft_predictions)
                # if epoch >=1:
                #     self.adaptive_weigth_adjustment(idx,epoch,50)
                self.local_model_states[idx]=copy.deepcopy(self.LocalModels[idx].model.state_dict())
                # del local_features
                # del local_soft_predictions
                gc.collect()


            # update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            global_features, global_soft_prediction = self.global_knowledge_aggregation(global_features,
                                                                                        global_soft_prediction)
            loss_avg = sum(local_losses) / len(local_losses)
            if epoch >= 1:
                # loss2_avg = sum(epoch_loss2) / len(epoch_loss2)
                loss3_avg = sum(epoch_loss3) / len(epoch_loss3)
                loss4_avg = sum(epoch_loss4) / len(epoch_loss4)
                G_avg = sum(epoch_G) / len(epoch_G)

                # wandb.log({'epoch': epoch, 'loss2_avg': loss2_avg})
                wandb.log({'epoch': epoch, 'loss3_avg': loss3_avg})
                wandb.log({'epoch': epoch, 'loss4_avg': loss4_avg})
                wandb.log({'epoch': epoch, 'G_avg': G_avg})

            train_loss.append(loss_avg)
            print("average loss:  ", loss_avg)
            print('average local test accuracy:', test_accuracy / self.args.num_clients)
            print('global test accuracy: ', self.global_test_accuracy())
            wandb.log({'epoch': epoch,
                       f'local test accuracy': test_accuracy / self.args.num_clients})
            wandb.log({'epoch': epoch,
                       f'global test accuracy': self.global_test_accuracy()})

        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()
