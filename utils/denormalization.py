def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
      single_img = True
      tensor = tensor[None,:,:,:]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret