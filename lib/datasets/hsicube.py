from torch.utils.data import Dataset


class hsicube(Dataset):
    def __init__(self, inputImageCube, inputLabelCube):
        super(hsicube, self).__init__()
        self.img = inputImageCube
        self.label = inputLabelCube
        self.len = inputImageCube.shape[0]

    def __getitem__(self, index):
        # outputImage = self.img[index][np.newaxis, :, :, :].transpose(0, 3, 1, 2)  # 3D2Dnet
        outputImage = self.img[index].transpose(2, 0, 1)  # restnet
        outputLabel = self.label[index]
        return outputImage, outputLabel

    def __len__(self):
        return self.len
