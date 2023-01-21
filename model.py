import torch
import torch.nn as nn
import torch.optim as optim


class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinear, self).__init__()
        # NonLinearクラスから見たコンストラクタ。恐らく省略可
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True), # 0未満0, 0以上はそのまま
            nn.BatchNorm1d(self.output_channels)) # バッチ正規化

    def forward(self, input_data): 
        return self.main(input_data)
        # nn.Sequentialに渡す定番の書き方
        # 上で定義しただけでは実行されない、このforward関数で実行


class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__() # NonLinearクラスから見たコンストラクタ
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)
        # 1つのcsvファイルには93行のデータあり
        # 93行の中の最大値抽出
        # 引き数は幾つの中の最大値か、幾つおきにスライドするか
        # 1dは列、2dは行

    def forward(self, input_data):
        out = input_data.view(-1, self.num_channels, self.num_points) # 三次元
        # num_channelsはMaxPoolクラスの入力で1024になっている
        # この時点でのoutのsize()は( , 1024, 93)
        out = self.main(out)
        # maxpoolしたことにより( , 1024, 1)?
        out = out.view(-1, self.num_channels)
        # ( , 1024)
        return out



class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(3, 64), 
            # 3はおそらくxyz。ここからスタート.input_data([930, 3])
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            # NonLinear(128, 128),
            # MaxPool(128, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9)
        )
        # 最後は(9x1)サイズのテンソルになっている。

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        # input_data ([930, 3])
        matrix = self.main(input_data).view(-1, 3, 3)
        # 
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        # 入力データと得られたアフィン行列の行列積を計算
        out = out.view(-1, 3)
        return out



class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            # NonLinear(128, 128),
            # MaxPool(128, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out



class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points), 
            # InputTnetの中にも非線形変換やMakPooling, 順伝搬がある
            # out = out.view(-1, 3)の形で出力
            NonLinear(3, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            # FeatureTNetの入力は64次元で固定
            # out = out.view(-1, 64)の形で出力
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels),
            )

    def forward(self, input_data):
        return self.main(input_data)