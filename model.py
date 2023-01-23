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
        out = input_data.view(-1, self.num_channels, self.num_points)
        # num_channelsはMaxPoolクラスの入力で1024になっている
        # スタートのinput_data([930, 3])、この層に入る直前に([930, 1024])
        # out.size() ([10 , 1024, 93])
        out = self.main(out)
        # maxpoolしたことにより(10 , 1024, 1)。
        # maxpoolの結果、1つのcsvファイルに対する特徴量が1024パターンの最大値になった？
        out = out.view(-1, self.num_channels)
        # ([10, 1024])
        return out



class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(3, 64), 
            # 3はおそらくxyz。ここからスタート.input_data([930, 3])
            NonLinear(64, 128),
            NonLinear(128, 1024), # ([930, 1024])
            MaxPool(1024, self.num_points),
            # view ([10 , 1024, 93])
            # maxpool1d ([10, 1024, 1])
            # view out ([10, 1024])
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9) # ([10, 9])
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        # input_data ([930, 3])
        matrix = self.main(input_data).view(-1, 3, 3)
        # ([10, 3, 3])
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        # 入力データと得られたアフィン行列の行列積を計算
        # torch.matmul(([10, 93, 3]), ([10, 3, 3]))
        # out # ([10, 93, 3])
        out = out.view(-1, 3) # ([930, 3])
        return out



class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            # input ([930, 64])?? input_data([930, 3])では??
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024), # ([930, 1024]) Input_TNetと同じ
            MaxPool(1024, self.num_points),
            # view ([10 , 1024, 93])
            # maxpool1d ([10, 1024, 1])
            # view out ([10, 1024])
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096) # ([10, 4096])
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data): # input_data([930, 3])??
        matrix = self.main(input_data).view(-1, 64, 64) 
        # matrix ([10, 64, 64])
        # input_data ([930, 64])
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        # ([10, 93, 64])
        out = out.view(-1, 64)
        # ([930, 64])
        return out



class PointNet(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points), 
            # InputTnetの中にも非線形変換やMaxPooling, 順伝搬がある
            # out ([930, 3])
            NonLinear(3, 64),
            NonLinear(64, 64), # ([930, 64])
            FeatureTNet(self.num_points),
            # FeatureTNetの入力層 (64, 64)
            # out ([930, 64])
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024), # ([930, 1024])同じ形
            MaxPool(1024, self.num_points),
            # out ([10, 1024])
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels), # ([10, 1])
            )

    def forward(self, input_data):
        return self.main(input_data)