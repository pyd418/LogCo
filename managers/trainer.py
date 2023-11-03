import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from relational_path.path_process import path_generate, path_emb_generate_batch, path_cross_loss, path_contrast_loss

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')      # 返回标量

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self, epoch):
        paths_epoch = 0
        paths_epoch_list = []
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        # num_rels = self.params.num_rels

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()       # model.train()
        model_params = list(self.graph_classifier.parameters())     # model参数list
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)

            # index = torch.LongTensor([0, 1]).to(device=self.params.device)
            rels_emb = self.graph_classifier.rel_emb
            rels_emb_gpu = rels_emb.weight

            # 获得路径正例，负例，表示等
            pos_paths, neg_paths, target_rels = path_generate(data_pos, self.params.max_path_len, self.params.num_rels)
            # pos_paths, neg_paths, target_rels = path_generate_mul_neg(data_pos, self.params.max_path_len, self.params.num_rels)
            # pos_paths_emb_batch, s_p_pos = path_emb_generate_batch(pos_paths, rels_emb.cpu(), target_rels.cpu())
            # neg_paths_emb_batch, s_p_neg = path_emb_generate_batch(neg_paths, rels_emb.cpu(), target_rels.cpu())

            pos_paths_emb_batch, s_p_pos = path_emb_generate_batch(pos_paths, rels_emb.cpu(), target_rels.cpu(), epoch, self.params.num_epochs, self.params.exp_dir, 0)
            neg_paths_emb_batch, s_p_neg = path_emb_generate_batch(neg_paths, rels_emb.cpu(), target_rels.cpu(), epoch, self.params.num_epochs, self.params.exp_dir, 0)

            # paths_nums = get_paths_nums(pos_paths)

            # 两个loss
            cross_loss = path_cross_loss(s_p_pos.to(device=self.params.device), rels_emb_gpu.to(device=self.params.device), target_rels)
            contrast_loss = path_contrast_loss(s_p_pos.to(device=self.params.device),
                                               s_p_neg.to(device=self.params.device),
                                               rels_emb.to(device=self.params.device), target_rels)
            # contrast_loss_2 = path_contrast_loss_2(pos_paths_emb_batch,
            #                                        neg_paths_emb_batch,
            #                                        rels_emb.to(device=self.params.device), target_rels, self.params.device)

            self.optimizer.zero_grad()      # 1) 清空过往梯度
            # score_pos = self.graph_classifier(data_pos)
            # score_neg = self.graph_classifier(data_neg)
            score_pos = self.graph_classifier(data_pos, s_p_pos.to(device=self.params.device))     # GraphClassifier.forward()
            score_neg = self.graph_classifier(data_neg, s_p_neg.to(device=self.params.device))
            loss_triple = self.criterion(score_pos, score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))       # 考虑到多个neg
            loss_1 = cross_loss.cpu().tolist()
            loss_2 = contrast_loss.cpu().tolist()
            loss = loss_triple + self.params.lambda_cross * loss_1 + self.params.lambda_contrast * loss_2
            # print(score_pos, score_neg, loss)
            loss.backward()                 # 2) 反向传播，计算当前梯度
            self.optimizer.step()           # 3) 根据梯度更新网络参数
            self.updates_counter += 1

            with torch.no_grad():           # 这一部分不track梯度, 为了使下面的计算图不占用内存
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()  # 所有得分函数, list拼接
                all_labels += targets_pos.tolist() + targets_neg.tolist()       # 所有labels, list拼接
                total_loss += loss          # 一个epoch的总loss
                # paths_epoch_list.append(paths_nums)

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:    # 利用valid 验证, 多次训练验证一次
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:       # 最高的auc
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        # paths_epoch = np.mean(paths_epoch_list)
        # logging.info(f"Average paths: {paths_epoch}.")

        auc = metrics.roc_auc_score(all_labels, all_scores)     # 机器学习准确率 roc面积
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))       # 权重参数的范数

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch(epoch)
            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))
            
            #     if result['auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        save_dir = os.path.join(self.params.exp_dir, f"{self.updates_counter}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.graph_classifier, os.path.join(save_dir,
                                                       'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')
