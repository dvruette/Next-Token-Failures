import torch
from transformers import AutoTokenizer
from tokenizing.numeral_tokenizer import NumeralTokenizer


class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        seqs = []
        loss_masks = []

        max_len = 0
        for prefix, target in data_list:
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            seq = torch.concatenate([prefix, target], dim=-1).long() + 1
            max_len = max(len(seq), max_len)
            seqs.append(seq)
            loss_mask = torch.cat([torch.zeros(len(prefix)), torch.ones(len(target))]).long()
            loss_masks.append(loss_mask)

        # add padding
        seqs = [torch.cat([seq, torch.zeros(max_len - len(seq)).long()]) for seq in seqs]
        loss_masks = [torch.cat([mask, torch.zeros(max_len - len(mask)).long()]) for mask in loss_masks]

        return seqs, loss_masks


def get_tokenizer(args):
    if args.model == 'gpt':
        t = NumeralTokenizer(args.num_nodes)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=args.num_nodes + 4, name='numeral')
    elif args.model.startswith('gpt2'):
        t = AutoTokenizer.from_pretrained('gpt2')
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50257 , name='gpt2')
    elif args.model.startswith('pythia'):
        t = AutoTokenizer.from_pretrained('EleutherAI/' + args.model)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50304, name='gpt2')
    elif args.model.startswith('phi'):
        t = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=51200, name='phi')

    return tokenizer
