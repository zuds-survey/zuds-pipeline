import sys
import time
import zuds
zuds.init_db()


# db.DBSession().autoflush = False
# db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on

BATCH_SIZE = 50
my_work = zuds.get_my_share_of_work(infile)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

for thumbids in batch(my_work, n=BATCH_SIZE):
    start = time.time()
    thumbs = zuds.DBSession().query(zuds.Thumbnail).filter(zuds.Thumbnail.id.in_(thumbids.tolist()))
    for t in thumbs:
        t.persist()
    stop = time.time()
    zuds.print_time(start, stop, t, 'get and persist')

    start = time.time()
    zuds.DBSession().commit()
    stop = time.time()
    zuds.print_time(start, stop, t, 'commit')


