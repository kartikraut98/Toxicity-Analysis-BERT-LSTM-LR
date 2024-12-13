
label_columns = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
thresholds = {
    'toxicity': 0.6000,
    'severe_toxicity': 0.0435,
    'obscene': 0.1042,
    'threat': 0.0878,
    'insult': 0.5000,
    'identity_attack': 0.1667,
    'sexual_explicit': 0.0189
}
data_paths = {
    'train': '../data/train_balanced.csv',
    'validation': '../data/val_balanced.csv',
    'test': '../data/test_balanced.csv'
}
