import torch.nn as nn
import torch

class Cnn(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.conv1 = self._make_block(in_channels=3, out_channels=8)
        self.conv2 = self._make_block(in_channels=8, out_channels=16)
        self.conv3 = self._make_block(in_channels=16, out_channels=32)
        self.conv4 = self._make_block(in_channels=32, out_channels=64)
        self.conv5 = self._make_block(in_channels=64, out_channels=128)

# Increasing out_channel: Số kênh đầu ra tăng dần để tăng khả năng biểu diễn và học các đặc trưng phức tạp hơn khi mạng sâu hơn.
# Ví dụ:
# Ở tầng đầu tiên: Nhận diện cạnh, góc.
# Ở tầng giữa: Nhận diện mắt, tai, hình dạng.
# Ở tầng cuối: Nhận diện khuôn mặt, vật thể.

        # Fully connected layer
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*7*7, out_features=512),  # 128*7*7: out_channels x feature map x feature map  || 8 x 8 <-> 512 / 2^6  -> 2^ <> max pooling
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)  # similar flatten | x.shape[1]*x.shape[2]*x.shape[3]  <=> -1

        x = self.fc1(x)
        return x

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(  # 1 block - usual has 5 block
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),  # out_channels (number of filters) is usually square of 2  - ex 2^3
            # padding: n - f + 1 * n - f + 1  --> output  | if has padding -> increase n --> size output = n ban dau
            nn.BatchNorm2d(num_features=out_channels),  # num_features = out_channels
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            # in_channels=8, out_channels=8 -> either same or increase double
            nn.BatchNorm2d(num_features=out_channels),  # num_features = out_channels
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)  # Max pooling
        )

if __name__ == '__main__':
    model = Cnn()
    input_data = torch.rand(16, 3, 224, 224)  # 512 / 2^6  -> 2^ <> max pooling  || if stride =1, padding = 0 then after conv decrease 2 unit: ex: in = 512 -> 510
    if torch.cuda.is_available():
        model.cuda()  # in_place function
        input_data = input_data.cuda()
    result = model(input_data)  # <=> call forward
    # print(result)
    print(result.shape)  # (16, 8, 222, 222) (B x C x H x W) : 8 -- out_channel : 222 -- kernel = 3 then decreasing 2 unit
    # if want output same size input then have 2 ways -> increase: padding = "same" |