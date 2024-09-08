import torch


class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        You do not have to modify this function, but you will need to implement the functions it calls.

        Args:
            x: list of lists of floats, data
            y: list of floats, labels
        """
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)

    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        return  torch.tensor(x, dtype=torch.float32).reshape((-1,len(x[0]))),  torch.tensor(y, dtype=torch.float32)

    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.mean(x, dim=0).reshape((-1,x.shape[1])), torch.std(x, dim=0).reshape((-1,x.shape[1]))

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        distance=  torch.norm(self.data_normalized - x, dim=1)
        idx = torch.argmin(distance)
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        distance=  torch.norm(self.data_normalized - x, dim=1)
        values, idxs = torch.topk(-distance,k)
        return self.data[idxs], self.label[idxs]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use the k-nearest neighbors of the input x to predict its regression label.
        The prediction will be the average value of the labels from the k neighbors.

        Args:
            x: 1D tensor [D]
            k: int, number of neighbors

        Returns:
            average value of labels from the k neighbors. Tensor of shape [1]
        """
        data, labs = self.get_k_nearest_neighbor(x,k)
        return torch.mean(labs)
