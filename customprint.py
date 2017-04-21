import time

start_time = time.time()

def print_time(label):
    print "----------------------------------------------------------"
    print label + ": " + str(time.time() - start_time) + " seconds"
    print "----------------------------------------------------------"