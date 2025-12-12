# tokenize titles
texts = list(test['title'].apply(lambda o: str(o)).values)
text_encodings = tokenizer(texts, 
                           padding=True, 
                           truncation=True, 
                           max_length=text_max_length)

test['input_ids'] = text_encodings['input_ids']
test['attention_mask'] = text_encodings['attention_mask']

del texts, text_encodings, tokenizer
_=gc.collect()


class Shopee(Dataset):
    def __init__(self, df, image_dir, augs):
        self.df = df
        self.augs = augs 
        self.image_dir = image_dir


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.df['input_ids'].iloc[idx]), 'attention_mask': torch.tensor(self.df['attention_mask'].iloc[idx])}
        
        # image
        image = cv2.imread(self.image_dir + self.df.loc[idx, 'image']).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augs(image=image)['image']

        return image, item
    
def make_aug(scale=1.0, horizontal_flip=False):
    im_size = int(round(scale*image_size))
    if horizontal_flip:
        valid_aug = A.Compose([A.LongestMaxSize(max_size=im_size, p=1.0),
                               A.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=0, p=1.0),
                               A.HorizontalFlip(p=1.0),
                               A.Normalize(p=1.0),
                               ToTensorV2(p=1.0)])
        
    else:
        valid_aug = A.Compose([A.LongestMaxSize(max_size=im_size, p=1.0),
                               A.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=0, p=1.0),
                               A.Normalize(p=1.0),
                               ToTensorV2(p=1.0)])
        
    return valid_aug

