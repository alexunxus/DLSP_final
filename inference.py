from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--task', type=str, help="task type: [default|SS]")

if __name__ == '__main__':
    task = argparser.task
    if task not in ['default', 'SS']:
        raise ValueError(f"Unknown task {task}")
    if task == 'default':
        # doing inference without self-supervised head
        pass
    else:
        # perform inference with self-supervision
        pass
