#coding=utf-8
import jittor as jt 
from jittor import nn
from models.vision_transformer import create_model
from models.utils import accuracy,AverageMeter
from dataset import create_val_dataset,create_train_dataset

import time

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def execute(self, x, target):
        logprobs = nn.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def main():

    jt.flags.use_cuda = 1

    model_name = 'vit_base_patch16_224'
    lr = 0.01
    train_dir = '/data/imagenet/train'
    eval_dir = '/data/imagenet/val'
    batch_size = 32
    input_size = 224
    num_workers = 4
    hflip = 0.5
    ratio = (0.75,1.3333333333333333)
    scale = (0.08,1.0)
    train_interpolation = 'random'
    num_epochs = 8

    model = create_model(model_name,pretrained=False,num_classes=1000)

    optimizer = jt.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True)

    loader_train = create_train_dataset(
        train_dir,
        img_size=input_size,
        batch_size=batch_size,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        interpolation=train_interpolation,
        num_workers=num_workers,
        shuffle = True
    )

    loader_eval = create_val_dataset(
        eval_dir,
        img_size=input_size,
        batch_size=1,
        num_workers=num_workers
    )

    
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    validate_loss_fn = nn.CrossEntropyLoss()


    try:
        for epoch in range(num_epochs):
            train_metrics = train_epoch(epoch, model, loader_train, optimizer, train_loss_fn)

            eval_metrics = validate(model, loader_eval, validate_loss_fn)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

def train_epoch(epoch, model, loader, optimizer, loss_fn, lr_scheduler=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        output = model(input)
        loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))

        optimizer.step(loss)

        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % 10 == 0:
            lr = optimizer.lr

            print(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0)/ batch_time_m.val,
                        rate_avg=input.size(0)/ batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with jt.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % 10 == 0):
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()