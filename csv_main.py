from model import *
from csv_sampler2 import *
import matplotlib as mpl
import matplotlib.pyplot as plt
# 拡張子を除いたファイル名（モジュールという）を読み込むことが可能
# するとsamler.data_sampler(2,3)のように関数や変数にアクセス可能


batch_size = 10
num_points = 93
num_labels = 1

x = []
y = []


def main():
    pointnet = PointNet(num_points, num_labels)

    new_param = pointnet.state_dict()
    # print(new_param)
    # 重みやバイアスと言ったパラメーターを抜き出してくれる
    new_param['main.0.main.6.bias'] = torch.eye(3, 3).view(-1)
    # torch.eye(3)と出力同じ。対角線上に1を配置して一次元化
    new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    pointnet.load_state_dict(new_param)
    # .load_state_dict()で渡すことが可能

    criterion = nn.BCELoss()
    # 損失関数
    optimizer = optim.Adam(pointnet.parameters(), lr=0.001)
    # 「optim最適化アルゴリズム（インスタンス化したモデル.paramteters(), 学習率）」

    loss_list = []
    accuracy_list = []

    for iteration in range(10000+1):

        pointnet.zero_grad()

        input_data, labels = csv_data_sampler()
        # input_data ([930, 3])
        # labels ([10, 1])

        output = pointnet(input_data)
        # output = pointnet.forward(input_data)　何故この形でない？
        # 基底クラスのcallの中に、y = self.forward(x) の形で入ってる？
        # output ([10, 1])

        output = nn.Sigmoid()(output)
        # シグモイド。(0, 0.5)で点対象となるS字曲線

        error = criterion(output, labels)
        error.backward()
        # 誤差逆伝搬

        optimizer.step()
        # 重み更新

        with torch.no_grad():
            output[output > 0.5] = 1 # テンソル[]　[]中の条件式を満たす要素を取り出し、変更している
            output[output < 0.5] = 0
            accuracy = (output==labels).sum().item()/batch_size

        loss_list.append(error.item())
        accuracy_list.append(accuracy)

        if iteration % 10 == 0:
            print('Iteration : {}   Loss : {}'.format(iteration, error.item()))
            print('Iteration : {}   Accuracy : {}'.format(iteration, accuracy))
            
        # x.append(iteration)
        # y.append(accuracy)
        # plt.plot(x, y, color="b")
        # plt.xlabel('Iteration')
        # plt.ylabel('Accuracy')
        # plt.show()
            
if __name__ == '__main__':
    main()