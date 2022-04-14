

def acc(x, y):
    '''
    x: (B, 1)
    y: (B, 1)
    '''
    x = x.view(-1)
    y = y.view(-1)
    return (x == y)/x.shape[0]
