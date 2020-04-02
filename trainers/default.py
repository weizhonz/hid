import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import updateScoreDiff, unfreeze_model_weights, freeze_model_weights
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    # batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    #l = [batch_time, data_time, losses, top1, top5]
    l = [losses, top1, top5]
    progress = ProgressMeter(
        len(train_loader),
        l,
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    image0, target0 = None, None
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # if i == 0:
        image0 = images
        target0 = target
        # measure data loading time
        # data_time.update(time.time() - end)

        if args.gpu is not None:
            image0 = image0.cuda(args.gpu, non_blocking=True)

        target0 = target0.cuda(args.gpu, non_blocking=True)
        l = 0
        a1 = 0
        a5 = 0
        for j in range(args.K):
            output = model(image0)
            loss = criterion(output, target0)
            acc1, acc5 = accuracy(output, target0, topk=(1, 5))
            l = l + loss
            a1 = a1 + acc1
            a5 = a5 + acc5
        l = l / args.K
        a1 = a1 / args.K
        a5 = a5 / args.K
        # measure accuracy and record loss
        # torch.Size([128, 3, 32, 32])
        # 128
        losses.update(l.item(), image0.size(0))
        top1.update(a1.item(), images.size(0))
        top5.update(a5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.conv_type != "SFESubnetConv":
            l.backward()
        else:
            updateScoreDiff(model, l)
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    # batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    #progress = ProgressMeter(
    #    len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    #)
    progress = ProgressMeter(
        len(val_loader), [losses, top1, top5], prefix="Test: "
    )
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
