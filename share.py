import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from utils import save_state_dict


def get_optimizer(opt, parameters, lr):
    if opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    return optimizer


def train_valid_split(dataset, val_size, batch_size):
    valid_size = int(len(dataset) * val_size)
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False, num_workers=4)
    return train_loader, valid_loader


def train_and_valid(model, criterion, optimizer,
                    train_iter, valid_iter,
                    epochs, early_threshold,
                    save_folder, best_model_name,
                    logger, device):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_valid_acc = 0
    epoch_counter = early_threshold
    logger.info("Start Training...")
    for epoch in range(1, epochs + 1):
        train_output = train_epoch(model, train_iter, criterion, optimizer, scheduler, device)
        train_loss.append(train_output['loss'])
        train_acc.append(train_output['acc'])

        valid_output = valid_epoch(model, valid_iter, criterion, device)
        valid_loss.append(valid_output['loss'])
        valid_acc.append(valid_output['acc'])

        logger.info(f"Epoch {epoch:>3d}: "
                    f"TRAIN loss {train_loss[-1]:>10.6f} acc {train_acc[-1]:>6.4f} | "
                    f"VALID loss {valid_loss[-1]:>10.6f} acc {valid_acc[-1]:>6.4f}")

        if valid_output['acc'] > best_valid_acc:
            best_valid_acc = valid_output['acc']
            epoch_counter = early_threshold
            save_state_dict(model, save_folder, best_model_name)
            logger.info("Saving Best...")
        else:
            epoch_counter -= 1

        if epoch_counter == 0:
            logger.info('Early Stopped!')
            break
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
    }


def train_epoch(model, data_iter, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_sample = 0
    for images, labels in data_iter:
        total_sample += len(labels)
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)
        total_loss += loss.sum().detach().cpu().item()

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        prediction = torch.argmax(output, 1)
        correct = (prediction == labels).sum().int().detach().cpu().item()
        total_correct += correct
    scheduler.step()

    epoch_loss = total_loss / total_sample
    epoch_acc = total_correct / total_sample
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }


def valid_epoch(model, data_iter, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_sample = 0

    with torch.no_grad():
        for images, labels in data_iter:
            total_sample += len(labels)
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)
            total_loss += loss.sum().cpu().item()

            prediction = torch.argmax(output, 1)
            correct = (prediction == labels).sum().int().cpu().item()
            total_correct += correct

    epoch_loss = total_loss / total_sample
    epoch_acc = total_correct / total_sample
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
    }