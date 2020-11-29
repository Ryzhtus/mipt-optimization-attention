import torch
from torch.utils.data import Dataset, DataLoader

class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, pad_to_max_length=True,
                                              return_attention_mask=True, return_tensors='pt', truncation=True)

        return {'review_text': review, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)}


def create_data_loader(data, tokenizer, max_len, batch_size):
    dataset = GPReviewDataset(reviews=data.content.to_numpy(), targets=data.sentiment.to_numpy(),
                              tokenizer=tokenizer, max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size, num_workers=4)
