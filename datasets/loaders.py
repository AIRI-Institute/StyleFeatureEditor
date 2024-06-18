from torch.utils.data import DataLoader


class InfiniteLoader(DataLoader):
    def __init__(
        self,
        *args,
        num_workers=0,
        pin_memory=True,
        is_infinite = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            multiprocessing_context="fork" if num_workers > 0 else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.dataset_iterator = super().__iter__()
        self.is_infinite = is_infinite

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            if self.is_infinite:
                x = next(self.dataset_iterator)
            else:
                raise StopIteration

        return x
