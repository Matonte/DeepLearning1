from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor]:
      return self.data.min(dim=1).values  , self.data.max(dim=1).values

    def find_the_largest_drop(self) -> torch.tensor:
       daily_averages = self.data.mean(dim=1)
       day_over_day_changes = daily_averages[1:] - daily_averages[:-1]
       largest_drop = day_over_day_changes.min()
       return largest_drop


    def find_the_most_extreme_day(self) -> torch.Tensor:
        daily_avg = self.data.mean(dim=1, keepdim=True)
        diff = torch.abs(self.data - daily_avg)
        max_diff_indices = diff.argmax(dim=1)
        most_extreme_values = self.data.gather(1, max_diff_indices.unsqueeze(1)).squeeze(1)
        return most_extreme_values

    
    def max_last_k_days(self, k: int) -> torch.Tensor:
        return self.data[-k:, :].max(dim=1)[0]
        

    def predict_temperature(self, k: int) -> torch.Tensor:
        return self.data[-k:, :].mean()


    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        differences = torch.sum(torch.abs(self.data - t), dim=1)
        closest_day_index = torch.argmin(differences)
        return closest_day_index
