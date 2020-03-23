import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import updateScore, unfreeze_model_weights, freeze_model_weights

__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
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
        if i == 0:
            image0 = images
            target0 = target
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            image0 = image0.cuda(args.gpu, non_blocking=True)

        target0 = target0.cuda(args.gpu, non_blocking=True)
        # compute output
        m_change = 50
        idx = 0
        while True:
            idx += 1
            m_change = int(40*((1000-idx) / 1000) + 10)
            output = model(image0)

            loss = criterion(output, target0)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target0, topk=(1, 5))
            # torch.Size([128, 3, 32, 32])
            # 128
            losses.update(loss.item(), image0.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            K = 100
            l = {}
            with torch.no_grad():
                for n, m in model.named_modules():
                    if hasattr(m, "mask"):
                        l[n] = m.mask.data.clone()

            while True:
                updateScore(model, args, K)
                output = model(image0)
                loss2 = criterion(output, target0)
                if loss2 < loss:
                    print("%d %.3f %.3f" % (K, loss.item(), loss2.item()))
                    break
                K = int(K*0.7)
                with torch.no_grad():
                    for n, m in model.named_modules():
                        if hasattr(m, "mask"):
                            m.mask.data.copy_(l[n])

                # output = model(image0)
                # loss3 = criterion(output, target0)

                print("%d %.3f %.3f" % (K, loss.item(), loss2.item()))

                if K == 1:
                    updateScore(model, args, m_change)
                    output = model(image0)
                    loss4 = criterion(output, target0)
                    print("%d %.3f %.3f %.3f" % (K, loss.item(), loss2.item(), loss4.item()))
                    break
            # optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                t = (num_batches * epoch + i) * batch_size
                progress.display(i)
                progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
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
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def modifier(args, epoch, model):
    return
