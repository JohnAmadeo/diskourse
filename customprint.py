import time

start_time = time.time()

def print_line(label=''):
    """
    -----------------------Label-----------------------
    """
    line_len = int(40 - len(label) / 2)
    print '-' * line_len + label + '-' * line_len

def print_time(label):
    """
    ----------------------------------------------
    Label: 0.00000 seconds
    ----------------------------------------------
    """
    print_line()
    print label + ": " + str(time.time() - start_time) + " seconds"
    print_line()

def print_blank(num=1):
    """
    \n
    """
    print ("\n" * num)
