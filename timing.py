# Andrew Lytle
# Dec 2020

# References:
# + https://blog.usejournal.com/
#   how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
# + http://hoardedhomelyhints.dietbuddha.com/
#   2012/12/52python-encapsulating-exceptions-with.html

import cProfile
import pstats
import time
from contextlib import contextmanager

@contextmanager
def timing():
    start = time.time()
    yield
    end = time.time()
    print("Elapsed time:", end-start, "seconds.")

@contextmanager
def profile():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats()
