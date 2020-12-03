#coding=utf-8
import torch
import time
import jittor as jt 
from jittor import nn
from models.vision_transformer import create_model
from models.utils import accuracy,AverageMeter
from dataset import create_val_dataset



def validate():
    bs=256
    # create model
    model  = create_model('vit_base_patch16_224',pretrained=True,num_classes=1000)
    criterion = nn.CrossEntropyLoss()
     
    dataset = create_val_dataset(root='/data/imagenet',batch_size=bs,num_workers=4,img_size=224)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with jt.no_grad():
        input = jt.random((bs,3,224,224))
        model(input)

        end=time.time()
        for batch_idx, (input, target) in enumerate(dataset):
            # dataset.display_worker_status()
            batch_size = input.shape[0]
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss, batch_size)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                # jt.sync_all(True)
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(dataset), batch_time=batch_time,
                        rate_avg=batch_size / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))
            
            # if batch_idx>50:break
            

    top1a, top5a = top1.avg, top5.avg
    top1=round(top1a, 4)
    top1_err=round(100 - top1a, 4)
    top5=round(top5a, 4)
    top5_err=round(100 - top5a, 4)

    print(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(top1,top1_err, top5,top5_err))


def main():
    jt.flags.use_cuda=1
    validate()

if __name__ == '__main__':
    main()