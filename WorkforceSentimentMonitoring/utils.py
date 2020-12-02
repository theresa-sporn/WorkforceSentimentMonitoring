import pandas as pd
import concurrent.futures
import functools
import time
from tqdm import tqdm
import time


def extract_negative(df):
    """return df with negative reviews and their labels"""
    df = df[['negatives']]
    df.loc[:,'sentiment'] = 0 # 0=> negative
    return df

def extract_positive(df):
    """return df with positive reviews and their labels"""
    df = df[['positives']]
    df.loc[:,'sentiment'] = 1 # 1=> positive
    return df

def progress_bar(expected_time, increments=10):

    def _progress_bar(func):

        def timed_progress_bar(future, expected_time, increments=10):
            """
            Display progress bar for expected_time seconds.
            Complete early if future completes.
            Wait for future if it doesn't complete in expected_time.
            """
            interval = expected_time / increments
            with tqdm(total=increments) as pbar:
                for i in range(increments - 1):
                    if future.done():
                        # finish the progress bar
                        # not sure if there's a cleaner way to do this?
                        pbar.update(increments - i)
                        return
                    else:
                        time.sleep(interval)
                        pbar.update()
                # if the future still hasn't completed, wait for it.
                future.result()
                pbar.update()

        @functools.wraps(func)
        def _func(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, *args, **kwargs)
                timed_progress_bar(future, expected_time, increments)

            return future.result()

        return _func

    return _progress_bar


def timing(f):
    """measures the time a function takes to complete"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

  
def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
