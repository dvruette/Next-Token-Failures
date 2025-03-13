import torch
from tqdm import tqdm

from utils.training_utils import AverageMeter


# Function to evaluate performance when generating
@torch.no_grad()
def evaluate(model, loader, ctx, temperature, top_k, results=None, mode='test'):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    # num_prefix_tokens = loader.dataset.num_prefix_tokens
    # num_target_tokens = loader.dataset.num_target_tokens

    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    total_acc = AverageMeter()
    tokens_corr: dict[int, AverageMeter] = {}
    path_corr: dict[int, AverageMeter] = {}
    bar = tqdm(loader)

    #model.set_cache(loader.dataset.device)
    for x, loss_mask in bar:
        # y = x[:, num_prefix_tokens:].clone()
        # x = x[:, :num_prefix_tokens].clone()

        x_ = x * (1 - loss_mask)
        prefix_lens = (x_ != 0).sum(dim=-1)
        total_lens = (x != 0).sum(dim=-1)

        # Group sequences by prefix length and path length
        groups = set(zip(prefix_lens.tolist(), total_lens.tolist()))

        for prefix_len, total_len in groups:
            ids = torch.where((prefix_lens == prefix_len) & (total_lens == total_len))[0]
            xi = x[ids, :prefix_len]
            num_target_tokens = total_len - prefix_len
            yi = x[ids, prefix_len : prefix_len + num_target_tokens]
            with ctx:
                y_pred = model.generate(xi, num_target_tokens, temperature=temperature, top_k=top_k)

            # Check how many tokens we get right and how many predictions are completely correct
            correct = yi.eq(y_pred[:, -num_target_tokens:]).float()

            # Completely correct
            completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
            total_acc.update(completely_correct.item(), len(ids))
            if num_target_tokens not in path_corr:
                path_corr[num_target_tokens] = AverageMeter()
            path_corr[num_target_tokens].update(completely_correct.item(), len(ids))

            # Individual token accuracy
            per_token_acc = correct.mean(dim=0)
            for j in range(num_target_tokens):
                if j not in tokens_corr:
                    tokens_corr[j] = AverageMeter()
                tokens_corr[j].update(per_token_acc[j].item(), len(ids))
        #model.reset_cache()

        bar.set_description(f'{mode} accuracy: {total_acc.get(percentage=True):.2f}')

    #model.empty_cache()

    # Switch back to train mode
    loader.dataset.train()
    model.train()

    if results is not None:
        results[mode + '/accuracy'] = total_acc.get(percentage=True)
        for i in tokens_corr.keys():
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
        for i in path_corr.keys():
            results[mode + '/path_' + str(i)] = path_corr[i].get(percentage=True)
    return results


# Function to evaluate performance when applying teacher forcing
@torch.no_grad()
def evaluate_forced(model, loader, ctx, results=None, mode='test'):
    """
    Generates sequences with teacher-forcing and calculates accuracies
    """
    # num_target_tokens = loader.dataset.num_target_tokens
    total_acc, total_loss = AverageMeter(), AverageMeter()
    # tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    for x, y in bar:
        # Produce logits with teacher-forcing (i.e. like during training)
        with ctx:
            logits, loss, accs = model(x, y)

        total_acc.update(val=accs['acc'], num=x.shape[0])
        total_loss.update(val=loss, num=x.shape[0])
        # for i in range(num_target_tokens):
        #     tokens_corr[i].update(accs['token_acc'], x.shape[0])

        bar.set_description('Forced Loss: {:.4f} Forced Acc: {:.2f}'.format(total_loss.get(),
                                                              total_acc.get(percentage=True)))

    if results is not None:
        results[mode + '/forced loss'] = total_loss.get()
        results[mode + '/forced accuracy'] = total_acc.get(percentage=True)
        # for i in range(num_target_tokens):
        #     results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)

    return results
