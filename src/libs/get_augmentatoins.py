def get_augmentations(ver: int):
    """To control augmentation version, we get acoustic augmentation list

    Args:
        ver (int): augmentation version
    Returns:
        List: Augmentation name list

    """
    if ver == 1:
        aug_list = []
    elif ver == 2:
        aug_list = ["white", "pink"]
    elif ver == 3:
        aug_list = ["white", "pink", "random_power", "upper_freq_decay"]
    elif ver == 4:
        aug_list = ["white", "pink", "random_power", "upper_freq_decay", "soundscape"]
    elif ver == 5:
        aug_list = [
            "white",
            "pink",
            "random_power",
            "upper_freq_decay",
            "soundscape",
            "reverberation",
        ]
    elif ver == 6:
        aug_list = [
            "white",
            "pink",
            "random_power",
            "upper_freq_decay",
            "soundscape",
            "reverberation",
            "time_mask",
            "freq_mask",
        ]
    elif ver == 7:
        aug_list = ['white', 'pink', 'random_power', 'upper_freq_decay', 'soundscape', 'time_mask', 'freq_mask']
    elif ver == 8:
        aug_list = ['white', 'pink', 'random_power', 'upper_freq_decay', 'soundscape', 'cutout']
    elif ver == 9:
        aug_list = ['white', 'pink', 'random_power', 'upper_freq_decay', 'soundscape', 'bandpass']
    elif ver == 10:
        aug_list = ["white", "pink", "random_power", "upper_freq_decay", "soundscape", 'esc50']
    else:
        message = "augmentation version not found"
        raise ValueError(message)
    return aug_list
