def get_augmentations(ver:int):
    """To control augmentation version, we get acoustic augmentation list

    Args:
        ver (int): augmentation version
    Returns:
        List: Augmentation name list
        
    """
    if ver == 1:
        aug_list = []
    elif ver == 2:
        aug_list = ['white', 'pink']
    elif ver == 3:
        aug_list = ['white', 'pink', 'random_power', 'upper_freq_decay']
    elif ver == 4:
        aug_list = ['white', 'pink', 'random_power', 'upper_freq_decay', 'soundscape']
    elif ver == 5:
        aug_list = ['white', 'pink', 'random_power', 'time_mask', 'freq_mask']
    else:
        message = "augmentation version not found"
        raise ValueError(message)
    return aug_list
