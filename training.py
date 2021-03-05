#!/usr/bin/env python3
import torch


def train(model, iterator, optimizer, criterion, clip, device):
    """
    Trains the model on the given data iterator.

    :param model: Model to be trained
    :param iterator: The data iterator
    :param optimizer: The optimizer to be used 
    :param criterion: The loss function
    :param clip: Max norm of the gradients
    :param device: CPU or GPU
    :return: Average train loss
    """
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        inputs, labels = batch['sample'], batch['sample']
        inputs = inputs.to(device)
        labels = labels.to(device)

        output, _ = model(inputs)

        # index of separator token
        idx = int(batch['separator_token'][0])

        # only consider loss on target
        shift_logits = output[..., idx:-1, :].contiguous()
        shift_labels = labels[..., idx + 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model on the given iterator data.

    :param model: Model to be evaluated
    :param iterator: The data iterator
    :param criterion: The loss function
    :param device:  CPU or GPU
    :return: Average evaluation loss
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            inputs, labels = batch['sample'], batch['sample']
            inputs = inputs.to(device)
            labels = labels.to(device)

            output, _ = model(inputs)

            # index of separator token
            idx = int(batch['separator_token'][0])

            # only consider loss on target
            shift_logits = output[..., idx:-1, :].contiguous()
            shift_labels = labels[..., idx + 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

