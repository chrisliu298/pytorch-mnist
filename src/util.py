def get_lr_schedule(lr_schedule_type):
    """Return a learning rate scheduler function."""
    if lr_schedule_type == "constant":
        return lambda t: 1
    elif lr_schedule_type == "inverse":  # for sgd, step wise
        return lambda t: 1 / (t + 1)
    elif lr_schedule_type == "inverse_slow":  # epoch wise
        return lambda t: 1 / (0.05 * t + 1)
    elif lr_schedule_type == "inverse_sqrt":  # for adam, epoch wise
        return lambda t: 1 / (t + 1) ** 0.5
