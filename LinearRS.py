import numpy as np
from model import Linear
import os


class LinearRS(Linear):

    def __init__(self, in_features, out_features, reg=1e-5):
        super().__init__(in_features, out_features, False, reg)
        self.name = 'Linear_RS_'

    def train(self,num_epochs, train_images, train_labels, val_images, val_labels):

        # train images (bn,bs,c,w,h)
        # train labels (bn,bs,1)
        # loss_best = float('inf')

        loss_lst=[]
        val_loss_lst=[]

        acc_lst=[]
        val_acc_lst=[]

        for epoch in range(num_epochs):
            loss_sum = 0.0
            acc_sum = 0.0
            for i, (x_train, y_train) in enumerate(zip(train_images, train_labels)):
                # x_train (bs,c*w*h)
                x_train = x_train.reshape(x_train.shape[0], -1)
                # y_one_hot (bs,out_features)
                y_one_hot = np.array([np.eye(self.out_features)[label] for label in y_train]).reshape((y_train.shape[0],-1))
                self.W_t = 0.001 * np.random.rand(self.in_features, self.out_features)
                loss, acc = self.evaluate(x_train, y_one_hot, 'train')
                loss_sum += loss
                acc_sum += acc

            self.save(f'{self.name}epoch_{epoch}.pt')
            file = f'{self.name}.txt'

            train_result = f'{self.mode} {epoch}, loss:{loss_sum / (i + 1)}, acc:{acc_sum / (i + 1)}\n'
            print(train_result)
            self.save_result(file, train_result)
            loss_lst.append(loss_sum / (i + 1))
            acc_lst.append(acc_sum / (i + 1))

            val_loss = 0.0
            val_acc = 0.0
            for j, (x_val, y_val) in enumerate(zip(val_images, val_labels)):
                x_val = x_val.reshape(x_val.shape[0], -1)
                # y_one_hot (bs,out_features)
                y_one_hot = np.array([np.eye(self.out_features)[label] for label in y_val]).reshape((y_val.shape[0],-1))
                loss, acc = self.evaluate(x_val, y_one_hot, 'valid')
                val_loss += loss
                val_acc += acc

            val_result = f'{self.mode} {epoch}, loss:{val_loss / (j + 1)}, acc:{val_acc / (j + 1)}\n'
            print(val_result)
            self.save_result(file, val_result)
            val_loss_lst.append(val_loss / (j + 1))
            val_acc_lst.append(val_acc / (j + 1))

        return loss_lst,acc_lst,val_loss_lst,val_acc_lst


if __name__=='__main__':
    import utils
    from model import Linear

    meta_file = 'batches.meta'

    label_names = utils.get_label_names(meta_file)
    train_images, train_labels = utils.load_train_data(200)

    val_images, val_labels = train_images[-1], train_labels[-1]
    val_images = val_images.reshape(1, val_images.shape[0], val_images.shape[1], val_images.shape[2], -1)
    val_labels = val_labels.reshape(1, val_labels.shape[0], -1)
    train_images, train_labels = train_images[:-1], train_labels[:-1]

    linear_rs = LinearRS(3072, 10, 10)

    linear_rs.train(train_images, train_labels, val_images, val_labels)
