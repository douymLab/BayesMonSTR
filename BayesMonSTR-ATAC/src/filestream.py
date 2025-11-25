import fcntl

def acquire_lock(file):
    try:
        fcntl.flock(file, fcntl.LOCK_EX)
        return True
    except BlockingIOError:
        return False
    
def release_lock(file):
    fcntl.flock(file, fcntl.LOCK_UN)